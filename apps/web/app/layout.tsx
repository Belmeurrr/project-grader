import type { Metadata } from 'next';
import { ClerkProvider } from '@clerk/nextjs';
import './globals.css';

export const metadata: Metadata = {
  title: 'Project Grader',
  description:
    'AI trading card grader and authenticator. Higher-accuracy grading and counterfeit detection for Pokemon, MTG, sports cards, and more.',
};

/**
 * Auth wiring: <ClerkProvider> mounts unconditionally because the
 * SDK is a no-op when NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is missing in
 * dev/test (we exit before calling getToken in lib/submission.ts). It
 * still renders children, so the unauth'd public cert page works.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className="min-h-screen antialiased">{children}</body>
      </html>
    </ClerkProvider>
  );
}
