import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sudoku AI — Deep Learning Solver",
  description: "Résolution de Sudoku par MLP, CNN, RNN, LSTM, GRU & architectures hybrides",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
