/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      { protocol: 'https', hostname: '*.s3.amazonaws.com' },
      { protocol: 'https', hostname: 'images.scryfall.com' },
      { protocol: 'https', hostname: 'images.pokemontcg.io' },
    ],
  },
  experimental: {
    typedRoutes: true,
  },
};

export default nextConfig;
