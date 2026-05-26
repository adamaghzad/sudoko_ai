"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import { Trophy, Zap, Target, BarChart3, AlertTriangle } from "lucide-react";
import type { SolveResponse, ModelName } from "@/lib/types";
import ModelCard from "./ModelCard";
import SudokuGrid from "./SudokuGrid";

interface SolverResultsProps {
  data: SolveResponse;
}

export default function SolverResults({ data }: SolverResultsProps) {
  const { puzzle, results, puzzle_valid, puzzle_error } = data;
  const [selected, setSelected] = useState<string>(
    () => Object.keys(results).find((k) => results[k as ModelName]?.is_valid) ?? "backtracking"
  );

  const entries = Object.entries(results) as [string, (typeof results)[ModelName]][];
  const validModels = entries.filter(([, r]) => r.is_valid);
  const fastestValid = validModels.sort((a, b) => a[1].time_ms - b[1].time_ms)[0];
  const selectedResult = results[selected as ModelName];

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Puzzle invalid warning */}
      {!puzzle_valid && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-3 rounded-2xl border border-amber-500/40 bg-amber-500/10 px-4 py-3"
        >
          <AlertTriangle className="w-5 h-5 shrink-0 text-amber-400" />
          <div>
            <p className="text-sm font-semibold text-amber-300">Puzzle invalide</p>
            <p className="text-xs text-amber-400/80">{puzzle_error} — aucun modèle ne peut résoudre un puzzle contradictoire.</p>
          </div>
        </motion.div>
      )}
      {/* Summary banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card rounded-2xl p-4 flex flex-wrap gap-4"
      >
        <Stat icon={<Target className="w-4 h-4 text-indigo-400" />}
              label="Indices" value={data.num_clues} />
        <Stat icon={<Trophy className="w-4 h-4 text-emerald-400" />}
              label="Solutions valides" value={`${validModels.length}/${entries.length}`} />
        {fastestValid && (
          <Stat icon={<Zap className="w-4 h-4 text-amber-400" />}
                label="Plus rapide" value={`${fastestValid[1].arch} (${fastestValid[1].time_ms}ms)`} />
        )}
        <Stat icon={<BarChart3 className="w-4 h-4 text-purple-400" />}
              label="Modèles DL" value={entries.length - 1} />
      </motion.div>

      {/* Main selected solution */}
      {selectedResult?.solution && (
        <motion.div
          key={selected}
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass-card rounded-2xl p-6 flex flex-col items-center gap-4"
        >
          <div className="text-center">
            <span
              className="text-xs font-bold uppercase tracking-widest"
              style={{ color: selectedResult.color }}
            >
              {selectedResult.arch}
            </span>
            <h3 className="text-white font-bold text-lg">Solution sélectionnée</h3>
            <p className="text-slate-400 text-sm">{selectedResult.desc}</p>
          </div>
          <SudokuGrid
            puzzle={puzzle}
            solution={selectedResult.solution}
            size="lg"
            color={selectedResult.color}
            highlightSolved
          />
          <div className="flex gap-4 text-sm">
            <span className="text-slate-400">
              Temps: <strong className="text-white">{selectedResult.time_ms} ms</strong>
            </span>
            <span className="text-slate-400">
              Confiance: <strong style={{ color: selectedResult.color }}>
                {Math.round(selectedResult.confidence * 100)}%
              </strong>
            </span>
          </div>
        </motion.div>
      )}

      {/* Model cards grid */}
      <div>
        <h3 className="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-3 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          Comparaison des architectures
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {entries.map(([name, result], i) => (
            <ModelCard
              key={name}
              name={name}
              result={result}
              puzzle={puzzle}
              index={i}
              isSelected={selected === name}
              onSelect={() => setSelected(name)}
              puzzleValid={puzzle_valid}
            />
          ))}
        </div>
      </div>

      {/* Comparison table */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="glass-card rounded-2xl overflow-hidden"
      >
        <div className="p-4 border-b border-slate-800">
          <h3 className="font-bold text-white">Tableau comparatif</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="text-left p-3">Architecture</th>
                <th className="text-right p-3">Temps (ms)</th>
                <th className="text-right p-3">Confiance</th>
                <th className="text-center p-3">Valide</th>
                <th className="text-center p-3">Entraîné</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([name, r]) => (
                <tr
                  key={name}
                  onClick={() => setSelected(name)}
                  className="border-b border-slate-800/50 hover:bg-white/02 cursor-pointer
                             transition-colors"
                  style={{
                    background: selected === name ? `${r.color}08` : undefined,
                  }}
                >
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ background: r.color }} />
                      <span className="font-semibold" style={{ color: r.color }}>{r.arch}</span>
                    </div>
                  </td>
                  <td className="p-3 text-right font-mono text-slate-300">
                    {r.trained ? `${r.time_ms}` : "—"}
                  </td>
                  <td className="p-3 text-right">
                    {r.trained ? (
                      <span style={{ color: r.color }}>
                        {Math.round(r.confidence * 100)}%
                      </span>
                    ) : "—"}
                  </td>
                  <td className="p-3 text-center">
                    {r.is_valid
                      ? "✅"
                      : !r.trained
                      ? "—"
                      : !puzzle_valid
                      ? "⚠️"
                      : "❌"}
                  </td>
                  <td className="p-3 text-center">
                    {r.trained ? "✅" : "⬜"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2">
      {icon}
      <span className="text-slate-400 text-sm">{label}:</span>
      <span className="font-bold text-white text-sm">{value}</span>
    </div>
  );
}
