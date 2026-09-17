/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export: the whole site becomes plain files that FastAPI serves
  // from the same origin as /api, so the LIFF page needs no CORS setup and
  // there is a single deployment to manage.
  output: "export",
  images: { unoptimized: true },   // no server = no on-demand optimiser
  trailingSlash: true,             // /diseases/ -> diseases/index.html
};

export default nextConfig;
