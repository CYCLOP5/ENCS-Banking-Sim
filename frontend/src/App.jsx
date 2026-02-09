import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";
// Eagerly preload topology data so the 3D graph is ready by the time
// the user navigates to the Simulation page.
import "./services/topologyCache";
import Methodology from "./pages/Methodology";
import Implementation from "./pages/Implementation";
import Simulation from "./pages/Simulation";
import BankExplorer from "./pages/BankExplorer";
import Benchmarking from "./pages/Benchmarking";
import Terminology from "./pages/Terminology";
import HowToUse from "./pages/HowToUse";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/methodology" element={<Methodology />} />
        <Route path="/implementation" element={<Implementation />} />
        <Route path="/simulation" element={<Simulation />} />
        <Route path="/benchmarking" element={<Benchmarking />} />
        <Route path="/banks" element={<BankExplorer />} />
        <Route path="/terminology" element={<Terminology />} />
        <Route path="/how-to-use" element={<HowToUse />} />
      </Route>
    </Routes>
  );
}
