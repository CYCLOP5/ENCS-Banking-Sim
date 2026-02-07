import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";
// Eagerly preload topology data so the 3D graph is ready by the time
// the user navigates to the Simulation page.
import "./services/topologyCache";
import Methodology from "./pages/Methodology";
import Simulation from "./pages/Simulation";
import BankExplorer from "./pages/BankExplorer";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/methodology" element={<Methodology />} />
        <Route path="/simulation" element={<Simulation />} />
        <Route path="/banks" element={<BankExplorer />} />
      </Route>
    </Routes>
  );
}
