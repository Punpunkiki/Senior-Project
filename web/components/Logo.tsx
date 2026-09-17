/**
 * Placeholder logo: a yellow durian with a green leaf and a small medical
 * cross. Drawn from scratch as geometry — it imitates no existing mark, and
 * is meant to be swapped for the real logo when one exists.
 */
export function Logo({ size = 36 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      role="img"
      aria-label="โลโก้หมอทุเรียน"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* fruit body */}
      <circle cx="32" cy="37" r="21" fill="#F2B705" />
      {/* thorns */}
      <g fill="#7A8B2E">
        {Array.from({ length: 12 }).map((_, i) => {
          const angle = (i / 12) * Math.PI * 2;
          const cx = 32 + Math.cos(angle) * 21;
          const cy = 37 + Math.sin(angle) * 21;
          const deg = (angle * 180) / Math.PI + 90;
          return (
            <polygon
              key={i}
              points={`${cx - 3.4},${cy} ${cx + 3.4},${cy} ${cx},${cy - 6.4}`}
              transform={`rotate(${deg} ${cx} ${cy})`}
            />
          );
        })}
      </g>
      {/* leaf */}
      <path
        d="M34 16c7-8 17-8 17-8s0 10-8 13c-5 2-9-5-9-5z"
        fill="#2E8B57"
      />
      {/* stem */}
      <rect x="30" y="13" width="4" height="9" rx="2" fill="#1F5F3A" />
      {/* medical cross */}
      <g fill="#FFFFFF">
        <rect x="28.5" y="29" width="7" height="17" rx="2" />
        <rect x="23.5" y="34" width="17" height="7" rx="2" />
      </g>
    </svg>
  );
}
