"use client";
import { motion } from "framer-motion";
import { CheckCircle, XCircle, Clock, Brain, AlertCircle, Target } from "lucide-react";
import clsx from "clsx";
import type { ModelResult } from "@/lib/types";
import SudokuGrid from "./SudokuGrid";

interface ModelCardProps {
  name: string;
  result: ModelResult;
  puzzle: number[];
  index: number;
  isSelected: boolean;
  onSelect: () => void;
  puzzleValid?: boolean;
}

export default function ModelCard({
  name,
  result,
  puzzle,
  index,
  isSelected,
  onSelect,
  puzzleValid = true,
}: ModelCardProps) {
  const confPct      = Math.round(result.confidence * 100);
  const confEmptyPct = Math.round((result.confidence_empty ?? result.confidence) * 100);
  const confMinPct   = Math.round((result.confidence_min  ?? result.confidence) * 100);
  const confMaxPct   = Math.round((result.confidence_max  ?? result.confidence) * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.08, type: "spring", stiffness: 200 }}
      onClick={onSelect}
      className={clsx(
        "glass-card rounded-2xl p-4 cursor-pointer transition-all duration-200",
        isSelected && "ring-2",
        !result.trained && "opacity-60"
      )}
      style={{
        ["--tw-ring-color" as string]: result.color,
        boxShadow: isSelected ? `0 0 20px ${result.color}30` : undefined,
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ background: result.color, boxShadow: `0 0 6px ${result.color}` }}
          />
          <span className="font-bold text-sm tracking-wide" style={{ color: result.color }}>
            {result.arch}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {!result.trained ? (
            <span className="text-xs text-slate-500 flex items-center gap-1">
              <AlertCircle className="w-3 h-3" /> Non entraîné
            </span>
          ) : result.is_valid ? (
            <CheckCircle className="w-4 h-4 text-emerald-400" />
          ) : (
            <XCircle className="w-4 h-4 text-red-400" />
          )}
        </div>
      </div>

      {/* Description */}
      <p className="text-xs text-slate-400 mb-3 leading-relaxed">{result.desc}</p>

      {/* Metrics */}
      {result.trained && (
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div className="bg-black/30 rounded-lg p-2 text-center">
            <div className="flex items-center justify-center gap-1 text-slate-400 text-xs mb-1">
              <Clock className="w-3 h-3" />
              Temps
            </div>
            <p className="text-sm font-bold text-white">{result.time_ms} ms</p>
          </div>
          <div className="bg-black/30 rounded-lg p-2 text-center">
            <div className="flex items-center justify-center gap-1 text-slate-400 text-xs mb-1">
              <Brain className="w-3 h-3" />
              Conf. moy.
            </div>
            <p className="text-sm font-bold" style={{ color: result.color }}>
              {confPct}%
            </p>
          </div>
          <div className="bg-black/30 rounded-lg p-2 text-center">
            <div className="flex items-center justify-center gap-1 text-slate-400 text-xs mb-1">
              <Target className="w-3 h-3" />
              Conf. vides
            </div>
            <p className="text-sm font-bold" style={{ color: result.color }}>
              {confEmptyPct}%
            </p>
          </div>
          <div className="bg-black/30 rounded-lg p-2 text-center">
            <div className="flex items-center justify-center gap-1 text-slate-400 text-xs mb-1">
              <span className="text-[10px] font-mono text-slate-400">min/max</span>
            </div>
            <p className="text-sm font-bold text-slate-300">
              <span className="text-red-400">{confMinPct}</span>
              <span className="text-slate-500 mx-0.5">–</span>
              <span className="text-emerald-400">{confMaxPct}</span>
              <span className="text-slate-500 text-xs">%</span>
            </p>
          </div>
        </div>
      )}

      {/* Confidence bar */}
      {result.trained && (
        <div className="bg-slate-800/50 rounded-full h-1 mb-3 overflow-hidden">
          <motion.div
            className="h-full rounded-full"
            style={{ background: result.color }}
            initial={{ width: 0 }}
            animate={{ width: `${confPct}%` }}
            transition={{ delay: index * 0.1 + 0.3, duration: 0.8, ease: "easeOut" }}
          />
        </div>
      )}

      {/* Mini grid */}
      {result.solution && isSelected && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="mt-3 flex justify-center"
        >
          <SudokuGrid
            puzzle={puzzle}
            solution={result.solution}
            size="sm"
            color={result.color}
          />
        </motion.div>
      )}

      {/* Status badge */}
      <div className="mt-2">
        {!result.trained ? (
          <span className="text-xs px-2 py-1 rounded-full bg-slate-800 text-slate-500">
            Poids non disponibles
          </span>
        ) : result.is_valid ? (
          <span className="text-xs px-2 py-1 rounded-full bg-emerald-900/40 text-emerald-400">
            Solution valide ✓
          </span>
        ) : !puzzleValid ? (
          <span className="text-xs px-2 py-1 rounded-full bg-amber-900/40 text-amber-400">
            Puzzle contradictoire
          </span>
        ) : name === "backtracking" ? (
          <span className="text-xs px-2 py-1 rounded-full bg-amber-900/40 text-amber-400">
            Aucune solution trouvée
          </span>
        ) : (
          <span className="text-xs px-2 py-1 rounded-full bg-orange-900/40 text-orange-400">
            Précision insuffisante
          </span>
        )}
      </div>
    </motion.div>
  );
}
