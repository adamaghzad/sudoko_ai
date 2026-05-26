"use client";
import { motion } from "framer-motion";
import { BookOpen } from "lucide-react";

const ARCHITECTURES = [
  {
    name: "MLP",
    color: "#6366f1",
    full: "Perceptron Multicouche",
    desc: "Données tabulaires. La grille 9×9 est aplatie en 810 features (one-hot) et passée à 4 couches denses avec BatchNorm + Dropout.",
    params: "~2.3M",
  },
  {
    name: "CNN",
    color: "#8b5cf6",
    full: "Réseau Convolutif",
    desc: "La grille est traitée comme une image 9×9×10. 6 blocs convolutifs (3×3) exploitent la structure spatiale des lignes, colonnes et boîtes.",
    params: "~1.8M",
  },
  {
    name: "RNN",
    color: "#ec4899",
    full: "Réseau Récurrent Simple",
    desc: "Les 81 cellules sont vues comme une séquence. RNN bidirectionnel avec embedding de dimension 16. Souffre de la disparition du gradient.",
    params: "~0.5M",
  },
  {
    name: "LSTM",
    color: "#f59e0b",
    full: "Long Short-Term Memory",
    desc: "3 portes (entrée, oubli, sortie) permettent une mémoire longue. Bidirectionnel + LayerNorm. Architecture de référence pour les séquences.",
    params: "~1.1M",
  },
  {
    name: "GRU",
    color: "#10b981",
    full: "Gated Recurrent Unit",
    desc: "2 portes (reset, update), 1 seul état caché. Plus léger que le LSTM tout en atteignant des performances comparables.",
    params: "~0.8M",
  },
  {
    name: "Hybrid",
    color: "#3b82f6",
    full: "CNN + LSTM",
    desc: "CNN extrait les features spatiales de la grille, puis un LSTM traite les lignes séquentiellement. Combine biais inductif spatial et temporel.",
    params: "~2.1M",
  },
];

export default function ArchitectureGuide() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card rounded-2xl p-6"
    >
      <h3 className="font-bold text-lg text-white flex items-center gap-2 mb-5">
        <BookOpen className="w-5 h-5 text-indigo-400" />
        Architectures Deep Learning
      </h3>
      <div className="space-y-3">
        {ARCHITECTURES.map((arch, i) => (
          <motion.div
            key={arch.name}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.07 }}
            className="rounded-xl p-3"
            style={{ background: `${arch.color}08`, border: `1px solid ${arch.color}20` }}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="font-bold text-sm px-2 py-0.5 rounded-md"
                    style={{ background: `${arch.color}20`, color: arch.color }}
                  >
                    {arch.name}
                  </span>
                  <span className="text-slate-400 text-xs">{arch.full}</span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">{arch.desc}</p>
              </div>
              <div className="text-right flex-shrink-0">
                <span className="text-xs text-slate-600 font-mono">{arch.params}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
