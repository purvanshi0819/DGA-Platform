import { Route, Routes } from "react-router-dom";
import AppLayout from "./layouts/AppLayout";
import Overview from "./pages/Overview";
import Diagnosis from "./pages/Diagnosis";
import Batch from "./pages/Batch";
import Compare from "./pages/Compare";
import PcaLab from "./pages/PcaLab";
import GasFeatures from "./pages/GasFeatures";
import DataQuality from "./pages/DataQuality";
import History from "./pages/History";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Overview />} />
        <Route path="diagnosis" element={<Diagnosis />} />
        <Route path="batch" element={<Batch />} />
        <Route path="compare" element={<Compare />} />
        <Route path="pca" element={<PcaLab />} />
        <Route path="features" element={<GasFeatures />} />
        <Route path="data" element={<DataQuality />} />
        <Route path="history" element={<History />} />
        <Route path="*" element={<div className="py-20 text-center text-slate-500">Page not found</div>} />
      </Route>
    </Routes>
  );
}
