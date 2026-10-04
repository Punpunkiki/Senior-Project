import Image from "next/image";

/**
 * The project logo: a durian wearing a lab coat and stethoscope.
 * Supplied as artwork; the white background was knocked out so the mark sits
 * on the green header as well as on the cream page.
 */
export function Logo({ size = 36 }: { size?: number }) {
  return (
    <Image
      src="/logo.png"
      alt="โลโก้หมอทุเรียน"
      width={size}
      height={size}
      priority={size > 60}
      style={{ width: size, height: size, objectFit: "contain" }}
    />
  );
}
