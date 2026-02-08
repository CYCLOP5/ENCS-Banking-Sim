#![allow(unused_variables, unused_assignments)]

use ndarray::{Array1, Array2, Axis};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

#[pyclass]
#[derive(Clone)]
struct NetworkState {

    w: Array2<f64>,

    external_assets: Array1<f64>,

    total_liabilities: Array1<f64>,

    total_assets: Array1<f64>,

    derivatives_exposure: Array1<f64>,

    asset_price: f64,

    equity: Array1<f64>,

    payments: Array1<f64>,

    n: usize,
}

#[pymethods]
impl NetworkState {

    #[new]
    fn new(
        w: PyReadonlyArray2<f64>,
        external_assets: PyReadonlyArray1<f64>,
        total_liabilities: PyReadonlyArray1<f64>,
        total_assets: PyReadonlyArray1<f64>,
        derivatives_exposure: PyReadonlyArray1<f64>,
    ) -> Self {
        let w = w.as_array().to_owned();
        let external_assets = external_assets.as_array().to_owned();
        let total_liabilities = total_liabilities.as_array().to_owned();
        let total_assets = total_assets.as_array().to_owned();
        let derivatives_exposure = derivatives_exposure.as_array().to_owned();
        let n = external_assets.len();
        let equity = &total_assets - &total_liabilities;
        let obligations = w.sum_axis(Axis(1));
        let payments = obligations;
        NetworkState {
            w,
            external_assets,
            total_liabilities,
            total_assets,
            derivatives_exposure,
            asset_price: 1.0,
            equity,
            payments,
            n,
        }
    }

    #[getter]
    fn get_asset_price(&self) -> f64 {
        self.asset_price
    }

    fn get_equity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.equity.clone().into_pyarray_bound(py)
    }

    fn get_payments<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.payments.clone().into_pyarray_bound(py)
    }

    fn get_external_assets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.external_assets.clone().into_pyarray_bound(py)
    }

    fn get_w<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.w.clone().into_pyarray_bound(py)
    }
}

#[pyclass]
#[derive(Clone)]
struct StepResult {
    #[pyo3(get)]
    t: usize,
    #[pyo3(get)]
    asset_price: f64,
    #[pyo3(get)]
    n_defaults: usize,
    #[pyo3(get)]
    n_distressed: usize,
    #[pyo3(get)]
    total_withdrawn: f64,
    #[pyo3(get)]
    fire_sale_volume: f64,
    #[pyo3(get)]
    failed_payments: usize,
    #[pyo3(get)]
    total_equity_loss: f64,
    #[pyo3(get)]
    margin_calls_total: f64,
    #[pyo3(get)]
    systemic_credit_losses: f64,
}

#[pymethods]
impl StepResult {
    fn __repr__(&self) -> String {
        format!(
            "StepResult(t={}, price={:.4}, defaults={}, distressed={}, withdrawn=${:.1}B, margin=${:.1}B, credit_loss=${:.1}B)",
            self.t,
            self.asset_price,
            self.n_defaults,
            self.n_distressed,
            self.total_withdrawn / 1e9,
            self.margin_calls_total / 1e9,
            self.systemic_credit_losses / 1e9,
        )
    }
}

#[pyfunction]
fn apply_shock(state: &mut NetworkState, trigger_idx: usize, severity: f64) {
    let loss = state.external_assets[trigger_idx] * severity;
    state.external_assets[trigger_idx] -= loss;
    state.total_assets[trigger_idx] -= loss;

    state.equity = &state.total_assets - &state.total_liabilities;
}

/// Internal implementation — takes a plain ndarray reference for initial_equity.
fn run_intraday_step_impl(
    state: &mut NetworkState,
    t: usize,
    initial_equity: &Array1<f64>,
    sigma: f64,
    panic_threshold: f64,
    alpha: f64,
    max_clearing_iter: usize,
    convergence_tol: f64,
    distress_threshold: f64,
    margin_sensitivity: f64,
) -> StepResult {
    let n = state.n;

    let solvency: Vec<f64> = (0..n)
        .map(|i| {
            if state.total_assets[i] > 0.0 {
                (state.total_assets[i] - state.total_liabilities[i]) / state.total_assets[i]
            } else {
                0.0
            }
        })
        .collect();

    let mut rng = StdRng::seed_from_u64(42 + t as u64);
    let normal = Normal::new(0.0, sigma).unwrap();

    let mut signals = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        for i in 0..n {
            signals[[j, i]] = solvency[i] + normal.sample(&mut rng);
        }
    }

    let mut total_withdrawn_per_bank = Array1::<f64>::zeros(n);
    let mut total_received_per_bank = Array1::<f64>::zeros(n);   

    let mut total_withdrawn_global = 0.0_f64;

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            if signals[[j, i]] < panic_threshold && state.w[[i, j]] > 0.0 {
                let withdrawn = state.w[[i, j]];
                total_withdrawn_per_bank[i] += withdrawn;  

                total_received_per_bank[j] += withdrawn;   

                total_withdrawn_global += withdrawn;
                state.w[[i, j]] = 0.0; 
            }
        }
    }

    let fire_sale_volume = total_withdrawn_global;

    let mut margin_calls_total = 0.0_f64;
    let mut systemic_credit_losses = 0.0_f64;

    if margin_sensitivity > 0.0 {
        let price_drop = 1.0 - state.asset_price; 

        if price_drop > 0.0 {
            for i in 0..n {
                let margin_call = state.derivatives_exposure[i] * price_drop * margin_sensitivity;
                if margin_call <= 0.0 {
                    continue;
                }
                margin_calls_total += margin_call;

                if state.external_assets[i] >= margin_call {

                    state.external_assets[i] -= margin_call;
                } else {

                    let shortfall = margin_call - state.external_assets[i];
                    state.external_assets[i] = 0.0;

                    systemic_credit_losses += shortfall;
                }
            }
        }
    }

    let volume_normalized = total_withdrawn_global / 1e12;
    state.asset_price *= (-alpha * volume_normalized).exp();
    let price = state.asset_price;

    for i in 0..n {

        state.external_assets[i] += total_received_per_bank[i];

        if total_withdrawn_per_bank[i] > 0.0 {
            let fire_cost = total_withdrawn_per_bank[i] / price; 

            state.external_assets[i] = (state.external_assets[i] - fire_cost).max(0.0);
        }

        state.total_liabilities[i] -= total_withdrawn_per_bank[i];

        state.total_assets[i] = state.external_assets[i] * price
            + state.w.column(i).sum(); 

    }

    state.equity = &state.total_assets - &state.total_liabilities;

    let obligations = state.w.sum_axis(Axis(1));

    let mut pi = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        if obligations[i] > 0.0 {
            for j in 0..n {
                pi[[i, j]] = state.w[[i, j]] / obligations[i];
            }
        }
    }

    let mut payments = obligations.clone();

    for _iter in 0..max_clearing_iter {
        let old_payments = payments.clone();

        let inflows = pi.t().dot(&payments);
        let wealth = &state.external_assets + &inflows;

        for i in 0..n {
            payments[i] = obligations[i].min(wealth[i].max(0.0));
        }

        let diff: f64 = (&payments - &old_payments).mapv(f64::abs).sum();
        if diff < convergence_tol {
            break;
        }
    }

    state.payments = payments.clone();

    let mut failed_payments: usize = 0;
    for i in 0..n {
        if obligations[i] > 1e-6 && (payments[i] / obligations[i]) < 0.999 {
            failed_payments += 1;
        }
    }

    let initial_eq = initial_equity;
    let mut n_defaults: usize = 0;
    let mut n_distressed: usize = 0;
    let mut total_equity_loss = 0.0_f64;

    for i in 0..n {
        if state.equity[i] < 0.0 {
            n_defaults += 1;
            total_equity_loss += state.equity[i].abs();
        } else {
            // Compare against pre-shock initial equity for distress detection
            if initial_eq[i] > 0.0 {
                let ratio = state.equity[i] / initial_eq[i];
                if ratio < distress_threshold {
                    n_distressed += 1;
                }
            }
        }
    }

    StepResult {
        t,
        asset_price: price,
        n_defaults,
        n_distressed,
        total_withdrawn: total_withdrawn_global,
        fire_sale_volume,
        failed_payments,
        total_equity_loss,
        margin_calls_total,
        systemic_credit_losses,
    }
}

/// Python-facing wrapper for run_intraday_step_impl.
#[pyfunction]
#[pyo3(signature = (state, t, initial_equity, sigma=0.05, panic_threshold=0.10, alpha=0.005,
                     max_clearing_iter=100, convergence_tol=1e-5,
                     distress_threshold=0.5, margin_sensitivity=1.0))]
fn run_intraday_step(
    state: &mut NetworkState,
    t: usize,
    initial_equity: PyReadonlyArray1<f64>,
    sigma: f64,
    panic_threshold: f64,
    alpha: f64,
    max_clearing_iter: usize,
    convergence_tol: f64,
    distress_threshold: f64,
    margin_sensitivity: f64,
) -> StepResult {
    let init_eq = initial_equity.as_array().to_owned();
    run_intraday_step_impl(
        state, t, &init_eq, sigma, panic_threshold, alpha,
        max_clearing_iter, convergence_tol, distress_threshold, margin_sensitivity,
    )
}

#[pyfunction]
#[pyo3(signature = (w, external_assets, total_liabilities, total_assets,
                     derivatives_exposure,
                     trigger_idx, severity, n_steps=10, sigma=0.05,
                     panic_threshold=0.10, alpha=0.005,
                     max_clearing_iter=100, convergence_tol=1e-5,
                     distress_threshold=0.5, margin_sensitivity=1.0,
                     circuit_breaker_threshold=0.0))]
fn run_full_simulation(
    py: Python<'_>,
    w: PyReadonlyArray2<f64>,
    external_assets: PyReadonlyArray1<f64>,
    total_liabilities: PyReadonlyArray1<f64>,
    total_assets: PyReadonlyArray1<f64>,
    derivatives_exposure: PyReadonlyArray1<f64>,
    trigger_idx: usize,
    severity: f64,
    n_steps: usize,
    sigma: f64,
    panic_threshold: f64,
    alpha: f64,
    max_clearing_iter: usize,
    convergence_tol: f64,
    distress_threshold: f64,
    margin_sensitivity: f64,
    circuit_breaker_threshold: f64,
) -> PyResult<Py<PyDict>> {

    let mut state = NetworkState::new(w, external_assets, total_liabilities, total_assets, derivatives_exposure);

    let initial_equity_saved: Vec<f64> = state.equity.to_vec();
    let initial_equity_arr = Array1::from(initial_equity_saved.clone());

    apply_shock(&mut state, trigger_idx, severity);

    let mut step_results: Vec<StepResult> = Vec::with_capacity(n_steps);
    let cb_floor = if circuit_breaker_threshold > 0.0 { 1.0 - circuit_breaker_threshold } else { 0.0 };
    let mut cb_triggered = false;
    let mut cb_step: usize = 0;

    for t in 1..=n_steps {
        // ── Circuit Breaker: if price has hit the floor, halt all trading ──
        if cb_floor > 0.0 && state.asset_price <= cb_floor {
            if !cb_triggered {
                cb_triggered = true;
                cb_step = t;
            }
            // Frozen step — no new withdrawals, price stays flat
            let n = state.n;
            let mut n_def: usize = 0;
            let mut n_dis: usize = 0;
            let mut eq_loss = 0.0_f64;
            for i in 0..n {
                if state.equity[i] < 0.0 {
                    n_def += 1;
                    eq_loss += state.equity[i].abs();
                } else if initial_equity_arr[i] > 0.0
                    && (state.equity[i] / initial_equity_arr[i]) < distress_threshold
                {
                    n_dis += 1;
                }
            }
            step_results.push(StepResult {
                t,
                asset_price: state.asset_price,
                n_defaults: n_def,
                n_distressed: n_dis,
                total_withdrawn: 0.0,
                fire_sale_volume: 0.0,
                failed_payments: 0,
                total_equity_loss: eq_loss,
                margin_calls_total: 0.0,
                systemic_credit_losses: 0.0,
            });
            continue;
        }

        let result = run_intraday_step_impl(
            &mut state,
            t,
            &initial_equity_arr,
            sigma,
            panic_threshold,
            alpha,
            max_clearing_iter,
            convergence_tol,
            distress_threshold,
            margin_sensitivity,
        );
        step_results.push(result);
    }

    let n = state.n;

    let mut status_strings: Vec<String> = Vec::with_capacity(n);
    let mut n_defaults_final: usize = 0;
    let mut n_distressed_final: usize = 0;

    for i in 0..n {
        if state.equity[i] < 0.0 {
            status_strings.push("Default".to_string());
            n_defaults_final += 1;
        } else if initial_equity_saved[i] > 0.0
            && (state.equity[i] / initial_equity_saved[i]) < distress_threshold
        {
            status_strings.push("Distressed".to_string());
            n_distressed_final += 1;
        } else {
            status_strings.push("Safe".to_string());
        }
    }

    let prices: Vec<f64> = step_results.iter().map(|s| s.asset_price).collect();
    let defaults_timeline: Vec<usize> = step_results.iter().map(|s| s.n_defaults).collect();
    let distressed_timeline: Vec<usize> = step_results.iter().map(|s| s.n_distressed).collect();
    let withdrawn_timeline: Vec<f64> = step_results.iter().map(|s| s.total_withdrawn).collect();
    let gridlock_timeline: Vec<usize> = step_results.iter().map(|s| s.failed_payments).collect();
    let equity_loss_timeline: Vec<f64> = step_results.iter().map(|s| s.total_equity_loss).collect();
    let margin_calls_timeline: Vec<f64> = step_results.iter().map(|s| s.margin_calls_total).collect();

    let systemic_credit_losses_total: f64 = step_results.iter().map(|s| s.systemic_credit_losses).sum();

    let total_equity_loss: f64 = initial_equity_saved.iter()
        .zip(state.equity.iter())
        .map(|(init, fin)| (init - fin).max(0.0))
        .sum();

    let dict = PyDict::new_bound(py);
    dict.set_item("n_steps", n_steps)?;
    dict.set_item("final_asset_price", state.asset_price)?;
    dict.set_item("n_defaults", n_defaults_final)?;
    dict.set_item("n_distressed", n_distressed_final)?;
    dict.set_item("equity_loss", total_equity_loss)?;
    dict.set_item("systemic_credit_losses", systemic_credit_losses_total)?;
    dict.set_item("status", status_strings)?;
    dict.set_item("final_equity", state.equity.to_vec())?;
    dict.set_item("initial_equity", initial_equity_saved)?;
    dict.set_item("payments", state.payments.to_vec())?;

    dict.set_item("price_timeline", prices)?;
    dict.set_item("defaults_timeline", defaults_timeline)?;
    dict.set_item("distressed_timeline", distressed_timeline)?;
    dict.set_item("withdrawn_timeline", withdrawn_timeline)?;
    dict.set_item("gridlock_timeline", gridlock_timeline)?;
    dict.set_item("equity_loss_timeline", equity_loss_timeline)?;
    dict.set_item("margin_calls_timeline", margin_calls_timeline)?;
    dict.set_item("circuit_breaker_triggered", cb_triggered)?;
    if cb_triggered {
        dict.set_item("circuit_breaker_step", cb_step)?;
    } else {
        dict.set_item("circuit_breaker_step", py.None())?;
    }

    Ok(dict.into())
}

#[pymodule]
fn encs_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NetworkState>()?;
    m.add_class::<StepResult>()?;
    m.add_function(wrap_pyfunction!(apply_shock, m)?)?;
    m.add_function(wrap_pyfunction!(run_intraday_step, m)?)?;
    m.add_function(wrap_pyfunction!(run_full_simulation, m)?)?;
    Ok(())
}

