"use client";
import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Brain, Camera, Pencil, Shuffle, Play, RotateCcw,
  Loader2, ChevronRight, Sparkles, Github
} from "lucide-react";
import clsx from "clsx";

import { solveGrid, solveImage, generatePuzzle } from "@/lib/api";
import type { SolveResponse, AppMode } from "@/lib/types";
import SudokuGrid from "@/components/SudokuGrid";
import ManualGrid from "@/components/ManualGrid";
import ImageUpload from "@/components/ImageUpload";
import SolverResults from "@/components/SolverResults";
import ModelsStatus from "@/components/ModelsStatus";
import ArchitectureGuide from "@/components/ArchitectureGuide";

const EMPTY_PUZZLE = Array(81).fill(0);

const DEMO_PUZZLE = [
  5,3,0, 0,7,0, 0,0,0,
  6,0,0, 1,9,5, 0,0,0,
  0,9,8, 0,0,0, 0,6,0,
  8,0,0, 0,6,0, 0,0,3,
  4,0,0, 8,0,3, 0,0,1,
  7,0,0, 0,2,0, 0,0,6,
  0,6,0, 0,0,0, 2,8,0,
  0,0,0, 4,1,9, 0,0,5,
  0,0,0, 0,8,0, 0,7,9,
];

const MODES: { id: AppMode; label: string; icon: React.ReactNode }[] = [
  { id: "manual",   label: "Grille manuelle", icon: <Pencil className="w-4 h-4" /> },
  { id: "image",    label: "Photo",           icon: <Camera className="w-4 h-4" /> },
  { id: "generate", label: "Aléatoire",       icon: <Shuffle className="w-4 h-4" /> },
];

export default function Home() {
  const [mode, setMode] = useState<AppMode>("manual");
  const [puzzle, setPuzzle] = useState<number[]>(DEMO_PUZZLE);
  const [result, setResult] = useState<SolveResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clues, setClues] = useState(30);

  const reset = () => {
    setResult(null);
    setError(null);
  };

  const handleSolveGrid = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await solveGrid(puzzle);
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Erreur de résolution");
    } finally {
      setLoading(false);
    }
  }, [puzzle]);

  const handleSolveImage = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await solveImage(file);
      setPuzzle(data.puzzle);
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Erreur d'analyse d'image");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await generatePuzzle(clues);
      setPuzzle(data.puzzle);
    } catch (e: any) {
      setError(e.message || "Erreur de génération");
    } finally {
      setLoading(false);
    }
  }, [clues]);

  const loadDemo = () => {
    setPuzzle(DEMO_PUZZLE);
    setResult(null);
    setError(null);
    setMode("manual");
  };

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-primary)" }}>
      {/* Background decoration */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full opacity-10"
             style={{ background: "radial-gradient(circle, #6366f1, transparent)" }} />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 rounded-full opacity-10"
             style={{ background: "radial-gradient(circle, #8b5cf6, transparent)" }} />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8">

        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-2xl bg-indigo-600/20 border border-indigo-500/40
                            flex items-center justify-center glow-indigo">
              <Brain className="w-6 h-6 text-indigo-400" />
            </div>
            <div className="text-left">
              <h1 className="text-3xl font-black text-white tracking-tight">
                Sudoku <span className="text-indigo-400">AI</span>
              </h1>
              <p className="text-slate-500 text-sm">Deep Learning Solver</p>
            </div>
          </div>
          <p className="text-slate-400 max-w-2xl mx-auto text-sm leading-relaxed">
            Résolution par <span className="text-indigo-300">6 architectures</span> de deep learning —
            MLP, CNN, RNN, LSTM, GRU & Hybride.
            Comparez les performances en temps réel.
          </p>
          <div className="flex items-center justify-center gap-2 mt-4">
            {["MLP","CNN","RNN","LSTM","GRU","Hybrid"].map((m) => (
              <span key={m} className="text-xs px-2 py-1 rounded-full bg-indigo-500/10
                                       border border-indigo-500/20 text-indigo-400">
                {m}
              </span>
            ))}
          </div>
        </motion.header>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

          {/* ── Left Panel ── */}
          <div className="xl:col-span-2 space-y-6">

            {/* Mode tabs */}
            <div className="flex gap-2 p-1 bg-slate-900/60 rounded-2xl border border-slate-800">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => { setMode(m.id); reset(); }}
                  className={clsx(
                    "flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl",
                    "text-sm font-semibold transition-all duration-200",
                    mode === m.id
                      ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/25"
                      : "text-slate-400 hover:text-white hover:bg-slate-800"
                  )}
                >
                  {m.icon}
                  {m.label}
                </button>
              ))}
            </div>

            {/* Input panel */}
            <motion.div
              key={mode}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="glass-card rounded-2xl p-6"
            >
              <AnimatePresence mode="wait">
                {mode === "manual" && (
                  <motion.div key="manual" className="space-y-5">
                    <div className="flex items-center justify-between">
                      <h2 className="font-bold text-white">Saisie manuelle</h2>
                      <button
                        onClick={loadDemo}
                        className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
                      >
                        <Sparkles className="w-3 h-3" />
                        Exemple
                      </button>
                    </div>
                    <p className="text-slate-500 text-xs">
                      Cliquez sur une cellule et tapez un chiffre (1-9). Suppr / 0 pour effacer.
                    </p>
                    <div className="flex justify-center">
                      <ManualGrid
                        puzzle={puzzle}
                        onChange={setPuzzle}
                        disabled={loading}
                      />
                    </div>
                    <div className="flex gap-3">
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => { setPuzzle(EMPTY_PUZZLE); reset(); }}
                        className="flex items-center gap-2 px-4 py-2.5 rounded-xl
                                   bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm
                                   transition-colors"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Effacer
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handleSolveGrid}
                        disabled={loading}
                        className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl
                                   bg-indigo-600 hover:bg-indigo-500 text-white font-semibold
                                   shadow-lg shadow-indigo-500/25 transition-all disabled:opacity-50
                                   disabled:cursor-not-allowed"
                      >
                        {loading ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        {loading ? "Résolution..." : "Résoudre avec tous les modèles"}
                        {!loading && <ChevronRight className="w-4 h-4 opacity-60" />}
                      </motion.button>
                    </div>
                  </motion.div>
                )}

                {mode === "image" && (
                  <motion.div key="image" className="space-y-4">
                    <div>
                      <h2 className="font-bold text-white mb-1">Résolution par image</h2>
                      <p className="text-slate-500 text-xs">
                        Prenez une photo d'un Sudoku. Le pipeline détecte la grille,
                        reconnaît les chiffres (DigitCNN) puis résout avec tous les modèles.
                      </p>
                    </div>
                    <ImageUpload onSolve={handleSolveImage} loading={loading} />
                    {result?.ocr && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="rounded-xl overflow-hidden border border-indigo-500/20"
                      >
                        <p className="text-xs text-slate-500 px-3 py-2 bg-slate-900/50">
                          Grille détectée par OpenCV
                        </p>
                        <img
                          src={`data:image/jpeg;base64,${result.ocr.grid_image_b64}`}
                          alt="Grille détectée"
                          className="w-full max-h-48 object-contain bg-black/40"
                        />
                      </motion.div>
                    )}
                  </motion.div>
                )}

                {mode === "generate" && (
                  <motion.div key="generate" className="space-y-5">
                    <div>
                      <h2 className="font-bold text-white mb-1">Génération aléatoire</h2>
                      <p className="text-slate-500 text-xs">
                        Génère un puzzle valide par backtracking avec N indices, puis le résout.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">Nombre d'indices</span>
                        <span className="font-bold text-indigo-300">{clues}</span>
                      </div>
                      <input
                        type="range"
                        min={17} max={50} value={clues}
                        onChange={(e) => setClues(+e.target.value)}
                        className="w-full accent-indigo-500"
                      />
                      <div className="flex justify-between text-xs text-slate-600">
                        <span>17 (difficile)</span>
                        <span>50 (facile)</span>
                      </div>
                    </div>

                    {puzzle.some((v) => v !== 0) && (
                      <div className="flex justify-center">
                        <SudokuGrid puzzle={puzzle} size="sm" color="#6366f1" />
                      </div>
                    )}

                    <div className="flex gap-3">
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handleGenerate}
                        disabled={loading}
                        className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl
                                   bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm
                                   transition-colors disabled:opacity-50"
                      >
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Shuffle className="w-4 h-4" />}
                        Générer
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handleSolveGrid}
                        disabled={loading || !puzzle.some((v) => v !== 0)}
                        className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl
                                   bg-indigo-600 hover:bg-indigo-500 text-white font-semibold
                                   shadow-lg shadow-indigo-500/25 transition-all disabled:opacity-50
                                   disabled:cursor-not-allowed"
                      >
                        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                        Résoudre
                      </motion.button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="rounded-xl p-4 bg-red-900/30 border border-red-500/30 text-red-400 text-sm"
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Results */}
            <AnimatePresence>
              {result && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <SolverResults data={result} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* ── Right Panel ── */}
          <div className="space-y-6">
            <ModelsStatus />
            <ArchitectureGuide />

            {/* Training command */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card rounded-2xl p-5"
            >
              <h3 className="font-bold text-white mb-3 flex items-center gap-2">
                <Play className="w-4 h-4 text-emerald-400" />
                Lancer l'entraînement
              </h3>
              <p className="text-slate-500 text-xs mb-3">
                Depuis le dossier <code className="text-indigo-400">backend/</code> :
              </p>
              <div className="space-y-2">
                <CodeBlock label="Mode rapide (test)" cmd="python -m training.train_all --quick" />
                <CodeBlock label="Entraînement complet" cmd="python -m training.train_all" />
                <CodeBlock label="Un seul modèle" cmd="python -m training.train_all --model lstm" />
                <CodeBlock label="Lancer l'API" cmd="python main.py" />
              </div>
            </motion.div>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-slate-600 text-xs">
          <p>Sudoku AI — EMSI Deep Learning Project 2025-2026</p>
          <p className="mt-1">MLP · CNN · RNN · LSTM · GRU · Hybrid · PyTorch · FastAPI · Next.js</p>
        </footer>
      </div>
    </div>
  );
}

function CodeBlock({ label, cmd }: { label: string; cmd: string }) {
  return (
    <div>
      <p className="text-xs text-slate-600 mb-1">{label}</p>
      <code className="block text-xs font-mono bg-black/40 border border-slate-800 rounded-lg
                       px-3 py-2 text-emerald-400 break-all">
        {cmd}
      </code>
    </div>
  );
}
