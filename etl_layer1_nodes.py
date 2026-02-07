import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import glob
warnings.filterwarnings('ignore')
BASE_PATH = Path(__file__).parent / "data"
FFIEC_DIR = BASE_PATH / "FEICR"
FFIEC_DATE = "09302025"
EBA_DIR = BASE_PATH / "EBAdata"
EU_METADATA = EBA_DIR / "TR_Metadata.xlsx"
BIS_LBS = BASE_PATH / "BIS" / "WS_LBS_D_PUB_csv_flat.csv"
def load_ffiec_schedule(schedule_name: str, date: str = FFIEC_DATE) -> pd.DataFrame:
    """Load a single FFIEC schedule"""
    pattern = f"FFIEC CDR Call Schedule {schedule_name} {date}*.txt"
    files = sorted(glob.glob(str(FFIEC_DIR / pattern)))
    if not files:
        print(f"    Warning: No files found for schedule {schedule_name}")
        return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep='\t', dtype=str, na_values=['', ' ', 'NA'], low_memory=False)
        df.columns = df.columns.str.strip().str.replace('"', '')
        dfs.append(df)
    if len(dfs) == 1:
        return dfs[0]
    result = dfs[0]
    for df in dfs[1:]:
        new_cols = [c for c in df.columns if c not in result.columns or c == 'IDRSSD']
        if 'IDRSSD' in new_cols:
            result = result.merge(df[new_cols], on='IDRSSD', how='outer')
    return result
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').fillna(0)
def get_col(df: pd.DataFrame, *col_names) -> pd.Series:
    for col in col_names:
        if col in df.columns:
            return safe_numeric(df[col])
    return pd.Series(0, index=df.index)

def load_eu_metadata() -> dict:
    """
    Load EU bank metadata from TR_Metadata.xlsx.
    Returns dict mapping LEI_Code -> Bank Name.
    """
    if not EU_METADATA.exists():
        print(f"    Warning: EU metadata file not found: {EU_METADATA}")
        return {}
    
    df = pd.read_excel(EU_METADATA, header=None, skiprows=4)
    lei_col = df.iloc[:, 2].astype(str).str.strip()
    name_col = df.iloc[:, 3].astype(str).str.strip()
    
    mapping = {}
    for lei, name in zip(lei_col, name_col):
        if lei and name and lei != 'nan' and name != 'nan' and len(lei) > 5:
            mapping[lei] = name
    
    print(f"    Loaded {len(mapping)} EU bank names from metadata")
    return mapping
def ingest_ffiec() -> pd.DataFrame:
    """
    Ingest US bank data from all relevant FFIEC Call.
    - Balance sheet (RC): Total Assets, Liabilities, Equity
    - Deposits (RCE): Deposit structure
    - Trading (RCD): Trading assets/liabilities
    - Derivatives (RCL): Notional values, fair values
    - Off-balance sheet (RCO): Commitments, guarantees
    """
    print("=" * 60)
    print("INGESTING FFIEC DATA (US BANKS) - ALL SCHEDULES")
    print("=" * 60)
    print("\n  Loading Schedule RC (Balance Sheet)...")
    df_rc = load_ffiec_schedule("RC")
    if df_rc.empty:
        raise FileNotFoundError("Schedule RC not found")
    print(f"    Loaded {len(df_rc)} banks")
    df_us = pd.DataFrame()
    df_us['bank_id'] = df_rc['IDRSSD'].astype(str)
    df_us['total_assets'] = get_col(df_rc, 'RCON2170', 'RCFD2170') * 1000  
    df_us['total_liabilities'] = get_col(df_rc, 'RCON2948', 'RCFD2948') * 1000
    df_us['equity_capital'] = get_col(df_rc, 'RCON3210', 'RCFD3210') * 1000
    df_us['total_deposits'] = get_col(df_rc, 'RCON2200', 'RCFN2200') * 1000
    df_us['trading_assets'] = get_col(df_rc, 'RCON3545', 'RCFD3545') * 1000
    df_us['trading_liabilities'] = get_col(df_rc, 'RCON3548', 'RCFD3548') * 1000
    df_us['securities_afs'] = get_col(df_rc, 'RCON1773', 'RCFD1773') * 1000
    df_us['securities_htm'] = get_col(df_rc, 'RCONJA22', 'RCFDJA22') * 1000
    df_us['loans_net'] = get_col(df_rc, 'RCONB529', 'RCFDB529') * 1000
    df_us['other_borrowed'] = get_col(df_rc, 'RCON3190', 'RCFD3190') * 1000
    print("  Loading Schedule RCE (Deposits)...")
    df_rce = load_ffiec_schedule("RCE")
    if not df_rce.empty and 'IDRSSD' in df_rce.columns:
        df_rce['IDRSSD'] = df_rce['IDRSSD'].astype(str)
        rce_cols = pd.DataFrame()
        rce_cols['bank_id'] = df_rce['IDRSSD']
        rce_cols['brokered_deposits'] = get_col(df_rce, 'RCON2365') * 1000
        rce_cols['transaction_accounts'] = get_col(df_rce, 'RCON2215') * 1000
        rce_cols['nontransaction_accounts'] = get_col(df_rce, 'RCON2385') * 1000
        df_us = df_us.merge(rce_cols, on='bank_id', how='left')
        print(f"    Merged deposit data")
    print("  Loading Schedule RCD (Trading Assets)...")
    df_rcd = load_ffiec_schedule("RCD")
    if not df_rcd.empty and 'IDRSSD' in df_rcd.columns:
        df_rcd['IDRSSD'] = df_rcd['IDRSSD'].astype(str)
        rcd_cols = pd.DataFrame()
        rcd_cols['bank_id'] = df_rcd['IDRSSD']
        rcd_cols['trading_treasury'] = get_col(df_rcd, 'RCON3531', 'RCFD3531') * 1000
        rcd_cols['trading_agency'] = get_col(df_rcd, 'RCON3532', 'RCFD3532') * 1000
        rcd_cols['trading_other'] = get_col(df_rcd, 'RCON3541', 'RCFD3541') * 1000
        df_us = df_us.merge(rcd_cols, on='bank_id', how='left')
        print(f"    Merged trading data")
    print("  Loading Schedule RCL (Derivatives)...")
    df_rcl = load_ffiec_schedule("RCL")
    if not df_rcl.empty and 'IDRSSD' in df_rcl.columns:
        df_rcl['IDRSSD'] = df_rcl['IDRSSD'].astype(str)
        rcl_cols = pd.DataFrame()
        rcl_cols['bank_id'] = df_rcl['IDRSSD']
        rcl_cols['deriv_ir_notional'] = (
            get_col(df_rcl, 'RCFDA126', 'RCONA126') +  
            get_col(df_rcl, 'RCFD8693', 'RCON8693') +  
            get_col(df_rcl, 'RCFD8694', 'RCON8694')    
        ) * 1000
        rcl_cols['deriv_fx_notional'] = (
            get_col(df_rcl, 'RCFDA127', 'RCONA127') +  
            get_col(df_rcl, 'RCFD8697', 'RCON8697')    
        ) * 1000
        rcl_cols['deriv_gross_positive_fv'] = (
            get_col(df_rcl, 'RCFD8741') +  
            get_col(df_rcl, 'RCFD8742') +  
            get_col(df_rcl, 'RCFD8743') +  
            get_col(df_rcl, 'RCFD8744')    
        ) * 1000
        rcl_cols['deriv_gross_negative_fv'] = (
            get_col(df_rcl, 'RCFD8745') +  
            get_col(df_rcl, 'RCFD8746') +  
            get_col(df_rcl, 'RCFD8747') +  
            get_col(df_rcl, 'RCFD8748')    
        ) * 1000
        rcl_cols['credit_deriv_sold'] = get_col(df_rcl, 'RCFDC968', 'RCONC968') * 1000
        rcl_cols['credit_deriv_bought'] = get_col(df_rcl, 'RCFDC969', 'RCONC969') * 1000
        df_us = df_us.merge(rcl_cols, on='bank_id', how='left')
        print(f"    Merged derivatives data")
    print("  Loading Schedule RCO (Off-Balance Sheet)...")
    df_rco = load_ffiec_schedule("RCO")
    if not df_rco.empty and 'IDRSSD' in df_rco.columns:
        df_rco['IDRSSD'] = df_rco['IDRSSD'].astype(str)
        rco_cols = pd.DataFrame()
        rco_cols['bank_id'] = df_rco['IDRSSD']
        rco_cols['unused_commitments'] = get_col(df_rco, 'RCON3814', 'RCFD3814') * 1000
        rco_cols['standby_loc'] = get_col(df_rco, 'RCON3819', 'RCFD3819') * 1000
        rco_cols['commercial_loc'] = get_col(df_rco, 'RCON3411', 'RCFD3411') * 1000
        rco_cols['securities_lent'] = get_col(df_rco, 'RCFDB981', 'RCONB981') * 1000
        rco_cols['securities_borrowed'] = get_col(df_rco, 'RCFDB980', 'RCONB980') * 1000
        df_us = df_us.merge(rco_cols, on='bank_id', how='left')
        print(f"    Merged off-balance sheet data")
    por_path = FFIEC_DIR / f"FFIEC CDR Call Bulk POR {FFIEC_DATE}.txt"
    if por_path.exists():
        print("  Loading bank names from POR...")
        df_por = pd.read_csv(por_path, sep='\t', dtype=str, low_memory=False)
        df_por.columns = df_por.columns.str.strip().str.replace('"', '')
        if 'IDRSSD' in df_por.columns and 'Financial Institution Name' in df_por.columns:
            names = df_por[['IDRSSD', 'Financial Institution Name']].copy()
            names.columns = ['bank_id', 'bank_name']
            df_us = df_us.merge(names, on='bank_id', how='left')
    if 'bank_name' not in df_us.columns:
        df_us['bank_name'] = 'US Bank ' + df_us['bank_id']
    df_us['region'] = 'US'
    numeric_cols = df_us.select_dtypes(include=[np.number]).columns
    df_us[numeric_cols] = df_us[numeric_cols].fillna(0)
    print(f"\n  Final US banks: {len(df_us)}")
    print(f"  Total US Assets: ${df_us['total_assets'].sum() / 1e12:.2f} trillion")
    print(f"  Columns loaded: {len(df_us.columns)}")
    return df_us
def ingest_eba() -> pd.DataFrame:
    """
    Ingest EU bank data from all EBA 
    - tr_cre.csv: Credit Risk exposures
    - tr_sov.csv: Sovereign exposures
    - tr_mrk.csv: Market Risk
    """
    print("\n" + "=" * 60)
    print("INGESTING EBA DATA (EU BANKS) - ALL FILES")
    print("=" * 60)
    print("\n  Loading tr_cre.csv (Credit Risk)...")
    cre_path = EBA_DIR / "tr_cre.csv"
    if not cre_path.exists():
        raise FileNotFoundError(f"EBA credit risk file not found: {cre_path}")
    df_cre = pd.read_csv(cre_path, dtype=str, low_memory=False)
    print(f"    Loaded {len(df_cre)} rows, {df_cre['LEI_Code'].nunique()} banks")
    banks = df_cre[['LEI_Code', 'NSA']].drop_duplicates()
    df_eu = banks.rename(columns={'LEI_Code': 'bank_id'})
    
    eu_names = load_eu_metadata()
    df_eu['bank_name'] = df_eu['bank_id'].map(eu_names)
    no_name = df_eu['bank_name'].isna()
    df_eu.loc[no_name, 'bank_name'] = df_eu.loc[no_name, 'bank_id'] + '_' + df_eu.loc[no_name, 'NSA']
    cre_exposure = df_cre[df_cre['Item'] == '2520501'].copy()
    cre_exposure['Amount'] = pd.to_numeric(cre_exposure['Amount'], errors='coerce')
    cre_agg = cre_exposure.groupby('LEI_Code')['Amount'].sum().reset_index()
    cre_agg.columns = ['bank_id', 'total_exposure_cre']
    cre_agg['total_exposure_cre'] = cre_agg['total_exposure_cre'] * 1e6  
    df_eu = df_eu.merge(cre_agg, on='bank_id', how='left')
    cre_rwa = df_cre[df_cre['Item'] == '2520521'].copy()
    cre_rwa['Amount'] = pd.to_numeric(cre_rwa['Amount'], errors='coerce')
    rwa_agg = cre_rwa.groupby('LEI_Code')['Amount'].sum().reset_index()
    rwa_agg.columns = ['bank_id', 'rwa_credit']
    rwa_agg['rwa_credit'] = rwa_agg['rwa_credit'] * 1e6
    df_eu = df_eu.merge(rwa_agg, on='bank_id', how='left')
    print(f"    Credit exposures: {df_eu['total_exposure_cre'].sum() / 1e12:.2f}T EUR")
    print("  Loading tr_sov.csv (Sovereign Exposures)...")
    sov_path = EBA_DIR / "tr_sov.csv"
    if sov_path.exists():
        df_sov = pd.read_csv(sov_path, dtype=str, low_memory=False)
        print(f"    Loaded {len(df_sov)} rows")
        sov_exposure = df_sov[df_sov['Item'] == '2520810'].copy()  
        sov_exposure['Amount'] = pd.to_numeric(sov_exposure['Amount'], errors='coerce')
        sov_agg = sov_exposure.groupby('LEI_Code')['Amount'].sum().reset_index()
        sov_agg.columns = ['bank_id', 'sovereign_exposure']
        sov_agg['sovereign_exposure'] = sov_agg['sovereign_exposure'] * 1e6
        df_eu = df_eu.merge(sov_agg, on='bank_id', how='left')
        print(f"    Sovereign exposures: {df_eu['sovereign_exposure'].sum() / 1e12:.2f}T EUR")
    print("  Loading tr_mrk.csv (Market Risk)...")
    mrk_path = EBA_DIR / "tr_mrk.csv"
    if mrk_path.exists():
        df_mrk = pd.read_csv(mrk_path, dtype=str, low_memory=False)
        print(f"    Loaded {len(df_mrk)} rows")
        mrk_exposure = df_mrk[df_mrk['Item'] == '2520401'].copy()
        mrk_exposure['Amount'] = pd.to_numeric(mrk_exposure['Amount'], errors='coerce')
        mrk_agg = mrk_exposure.groupby('LEI_Code')['Amount'].sum().reset_index()
        mrk_agg.columns = ['bank_id', 'market_risk_rwa']
        mrk_agg['market_risk_rwa'] = mrk_agg['market_risk_rwa'] * 1e6
        df_eu = df_eu.merge(mrk_agg, on='bank_id', how='left')
        print(f"    Market risk RWA: {df_eu['market_risk_rwa'].sum() / 1e12:.2f}T EUR")
    df_eu['total_assets'] = df_eu['total_exposure_cre'].fillna(0)  
    df_eu['total_liabilities'] = df_eu['total_assets'] * 0.95  
    df_eu['equity_capital'] = df_eu['total_assets'] * 0.05
    df_eu['region'] = 'EU'
    numeric_cols = df_eu.select_dtypes(include=[np.number]).columns
    df_eu[numeric_cols] = df_eu[numeric_cols].fillna(0)
    print(f"\n  Final EU banks: {len(df_eu)}")
    print(f"  Total EU Assets: €{df_eu['total_assets'].sum() / 1e12:.2f} trillion")
    print(f"  Columns loaded: {len(df_eu.columns)}")
    return df_eu
def ingest_bis(chunk_size: int = 100000, max_chunks: int = 400) -> pd.DataFrame:
    """
    Ingest BIS Locational Banking Statistics for country-level aggregates.
    """
    print("\n" + "=" * 60)
    print("INGESTING BIS DATA (COUNTRY AGGREGATES)")
    print("=" * 60)
    if not BIS_LBS.exists():
        print(f"  Warning: BIS file not found: {BIS_LBS}")
        return pd.DataFrame(columns=['country', 'total_claims', 'total_liabilities'])
    print(f"  Loading BIS LBS (chunked, chunksize={chunk_size})...")
    aggregates = []
    chunks_processed = 0
    try:
        for chunk in pd.read_csv(BIS_LBS, chunksize=chunk_size, dtype=str, low_memory=False):
            chunks_processed += 1
            col_map = {}
            for col in chunk.columns:
                short = col.split(':')[0]
                col_map[col] = short
            chunk = chunk.rename(columns=col_map)
            if 'L_MEASURE' in chunk.columns and 'L_POSITION' in chunk.columns:
                mask = chunk['L_MEASURE'].str.contains('S:', na=False)
                mask &= (chunk['L_POSITION'].str.contains('C:', na=False) | 
                         chunk['L_POSITION'].str.contains('L:', na=False))
                filtered = chunk[mask].copy()
                if len(filtered) > 0:
                    filtered['OBS_VALUE'] = pd.to_numeric(filtered['OBS_VALUE'], errors='coerce')
                    filtered['pos_type'] = filtered['L_POSITION'].str[0]
                    aggregates.append(filtered[['L_REP_CTY', 'pos_type', 'OBS_VALUE']])
            if chunks_processed >= max_chunks:
                print(f"    Reached max chunks ({max_chunks}), stopping...")
                break
            if chunks_processed % 50 == 0:
                print(f"    Processed {chunks_processed} chunks...")
    except Exception as e:
        print(f"  Warning: Error reading BIS file: {e}")
        return pd.DataFrame(columns=['country', 'total_claims', 'total_liabilities'])
    if not aggregates:
        print("  No matching BIS data found")
        return pd.DataFrame(columns=['country', 'total_claims', 'total_liabilities'])
    df_bis = pd.concat(aggregates, ignore_index=True)
    print(f"  Filtered rows: {len(df_bis)}")
    country_agg = df_bis.groupby(['L_REP_CTY', 'pos_type'])['OBS_VALUE'].sum().reset_index()
    country_pivot = country_agg.pivot(
        index='L_REP_CTY', columns='pos_type', values='OBS_VALUE'
    ).reset_index()
    country_pivot.columns.name = None
    country_pivot = country_pivot.rename(columns={
        'L_REP_CTY': 'country',
        'C': 'total_claims',
        'L': 'total_liabilities'
    })
    for col in ['total_claims', 'total_liabilities']:
        if col in country_pivot.columns:
            country_pivot[col] = country_pivot[col].fillna(0) * 1e6  
    print(f"  Final countries with data: {len(country_pivot)}")
    return country_pivot
def create_master_nodes_df(df_us: pd.DataFrame, df_eu: pd.DataFrame) -> pd.DataFrame:
    """Merge US and EU dataframes into master nodes dataframe."""
    print("\n" + "=" * 60)
    print("CREATING MASTER NODES DATAFRAME")
    print("=" * 60)
    common_cols = ['bank_id', 'bank_name', 'total_assets', 'total_liabilities', 'equity_capital', 'region']
    for col in common_cols:
        if col not in df_us.columns:
            df_us[col] = 0 if col not in ['bank_id', 'bank_name', 'region'] else ''
        if col not in df_eu.columns:
            df_eu[col] = 0 if col not in ['bank_id', 'bank_name', 'region'] else ''
    master_df = pd.concat([df_us, df_eu], ignore_index=True, sort=False)
    print(f"  Combined records: {len(master_df)}")
    print(f"  US banks: {len(df_us)}")
    print(f"  EU banks: {len(df_eu)}")
    master_df = master_df.dropna(subset=['bank_id'])
    master_df = master_df[master_df['bank_id'] != '']
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    master_df[numeric_cols] = master_df[numeric_cols].fillna(0)
    if 'leverage_ratio' not in master_df.columns:
        master_df['leverage_ratio'] = np.where(
            master_df['equity_capital'] > 0,
            master_df['total_assets'] / master_df['equity_capital'],
            0
        )
    master_df = master_df.sort_values('total_assets', ascending=False).reset_index(drop=True)
    print(f"  Final columns: {len(master_df.columns)}")
    print(f"  Total banks: {len(master_df)}")
    print(f"  Total assets: ${master_df['total_assets'].sum() / 1e12:.2f} trillion")
    return master_df
def main():
    """Main ETL pipeline."""
    print("\n" + "=" * 60)
    print("ENCS LAYER 1: COMPLETE INPUT & NORMALIZATION ETL")
    print("Systemic Risk Engine - Full Data Ingestion")
    print("=" * 60)
    try:
        df_us = ingest_ffiec()
    except Exception as e:
        print(f"ERROR ingesting FFIEC: {e}")
        import traceback; traceback.print_exc()
        df_us = pd.DataFrame()
    try:
        df_eu = ingest_eba()
    except Exception as e:
        print(f"ERROR ingesting EBA: {e}")
        import traceback; traceback.print_exc()
        df_eu = pd.DataFrame()
    try:
        country_aggregates_df = ingest_bis()
    except Exception as e:
        print(f"ERROR ingesting BIS: {e}")
        country_aggregates_df = pd.DataFrame()
    if not df_us.empty or not df_eu.empty:
        master_nodes_df = create_master_nodes_df(df_us, df_eu)
    else:
        master_nodes_df = pd.DataFrame()
    print("\n" + "=" * 60)
    print("FINAL OUTPUT VERIFICATION")
    print("=" * 60)
    if not master_nodes_df.empty:
        print("\n>>> master_nodes_df.head(10):")
        print(master_nodes_df.head(10).to_string())
        print("\n>>> master_nodes_df.info():")
        master_nodes_df.info()
        print("\n>>> All columns:")
        print(list(master_nodes_df.columns))
        print("\n>>> Summary by Region:")
        region_summary = master_nodes_df.groupby('region').agg({
            'bank_id': 'count',
            'total_assets': 'sum',
            'total_liabilities': 'sum',
            'equity_capital': 'sum'
        }).rename(columns={'bank_id': 'bank_count'})
        print(region_summary)
    if not country_aggregates_df.empty:
        print("\n>>> country_aggregates_df.head(10):")
        print(country_aggregates_df.head(10).to_string())
        print(f"\n    Total countries: {len(country_aggregates_df)}")
    OUTPUT_DIR = BASE_PATH / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print("SAVING OUTPUT FILES")
    print("=" * 60)
    if not master_nodes_df.empty:
        master_path = OUTPUT_DIR / "master_nodes.csv"
        master_nodes_df.to_csv(master_path, index=False)
        print(f"  Saved: {master_path}")
        print(f"    → {len(master_nodes_df)} banks × {len(master_nodes_df.columns)} columns")
    if not country_aggregates_df.empty:
        country_path = OUTPUT_DIR / "country_aggregates.csv"
        country_aggregates_df.to_csv(country_path, index=False)
        print(f"  Saved: {country_path}")
        print(f"    → {len(country_aggregates_df)} countries")
    print("\n" + "=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    return master_nodes_df, country_aggregates_df
if __name__ == "__main__":
    master_nodes_df, country_aggregates_df = main()