"use client";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle, XCircle, TrendingUp, Database, ChevronDown, Clock } from "lucide-react";
import { getModelsStatus } from "@/lib/api";
import type { ModelStatus, ModelName } from "@/lib/types";
import TrainingChart from "./TrainingChart";

export default function ModelsStatus() {
  const [status, setStatus] = useState<Record<ModelName, ModelStatus> | null>(null);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    getModelsStatus()
      .then(setStatus)
      .catch(() => setStatus(null))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="glass-card rounded-2xl p-6 animate-pulse">
        <div className="h-4 bg-slate-700 rounded w-1/3 mb-4" />
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-12 bg-slate-700/50 rounded-xl mb-2" />
        ))}
      </div>
    );
  }

  if (!status) {
    return (
      <div className="glass-card rounded-2xl p-6 text-center text-slate-500">
        <p>API non disponible — lancez le backend.</p>
        <code className="block mt-2 text-xs font-mono bg-black/30 rounded px-3 py-2 text-indigo-400">
          cd backend && python main.py
        </code>
      </div>
    );
  }

  const models = Object.entries(status) as [ModelName, ModelStatus][];
  const trainedCount = models.filter(([, s]) => s.trained).length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-6"
    >
      <div className="flex items-center justify-between mb-5">
        <h3 className="font-bold text-lg text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-indigo-400" />
          État des modèles
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-400">{trainedCount}/{models.length}</span>
          <span className={`text-xs px-2 py-0.5 rounded-full ${
            trainedCount === models.length
              ? "bg-emerald-900/40 text-emerald-400"
              : trainedCount > 0
              ? "bg-amber-900/40 text-amber-400"
              : "bg-slate-800 text-slate-500"
          }`}>
            {trainedCount === models.length ? "Tous prêts" : trainedCount > 0 ? "Partiel" : "Non entraîné"}
          </span>
        </div>
      </div>

      <div className="space-y-2">
        {models.map(([name, s], i) => (
          <div key={name}>
            <motion.button
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.06 }}
              onClick={() => s.history && setExpanded(expanded === name ? null : name)}
              className="w-full flex items-center gap-3 p-3 rounded-xl text-left"
              style={{ background: `${s.color}10`, border: `1px solid ${s.color}20` }}
            >
              <div
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ background: s.color, boxShadow: `0 0 6px ${s.color}` }}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm" style={{ color: s.color }}>{s.arch}</span>
                  {s.history && (
                    <span className="text-xs text-slate-500 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      {s.history.epochs_trained} ép.
                      {s.history.best_epoch !== undefined && (
                        <span className="text-slate-600">(best #{s.history.best_epoch})</span>
                      )}
                    </span>
                  )}
                  {s.history?.best_val_puzzle_acc !== undefined && (
                    <span className="text-xs text-emerald-500">
                      {(s.history.best_val_puzzle_acc * 100).toFixed(1)}% puzz.
                    </span>
                  )}
                  {s.history?.train_time_sec !== undefined && (
                    <span className="text-xs text-slate-600 flex items-center gap-0.5">
                      <Clock className="w-3 h-3" />{s.history.train_time_sec}s
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-500 truncate">{s.desc}</p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {s.weight_exists && (
                  <span className="text-xs text-slate-600">{s.weight_size_mb}MB</span>
                )}
                {s.trained ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400" />
                ) : (
                  <XCircle className="w-4 h-4 text-slate-600" />
                )}
                {s.history && (
                  <ChevronDown
                    className="w-3 h-3 text-slate-600 transition-transform"
                    style={{ transform: expanded === name ? "rotate(180deg)" : "none" }}
                  />
                )}
              </div>
            </motion.button>

            <AnimatePresence>
              {expanded === name && s.history && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="pt-2 pb-1">
                    <TrainingChart
                      history={s.history}
                      color={s.color}
                      modelName={s.arch}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>

      {trainedCount === 0 && (
        <div className="mt-4 p-3 rounded-xl bg-amber-900/20 border border-amber-500/30 text-amber-400 text-sm">
          Aucun modèle entraîné. Lancez depuis le dossier <code className="text-white">backend/</code> :
          <code className="block mt-1 text-xs font-mono bg-black/30 rounded px-2 py-1.5">
            python -m training.train_all --quick
          </code>
        </div>
      )}
    </motion.div>
  );
}
