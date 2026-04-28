import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Project Grader',
  description:
    'AI trading card grader and authenticator. Higher-accuracy grading and counterfeit detection for Pokemon, MTG, sports cards, and more.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
