<script>
  let { size = "100%" } = $props();
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 32 32"
  fill="none"
  xmlns="http://www.w3.org/2000/svg"
  class="cat-face-svg"
>
  <path
    d="M22 3.00006C24.5005 2.97976 26 7.99749 26 7.99749C28.6835 10.6843 31 14.0638 31 17.9923C31 26.0133 23.0907 29.2354 16 28.9867C8.90926 29.2354 1 26.0133 1 17.9923C1 14.0638 3.31647 10.6843 6 7.99749C6 7.99749 7.49951 2.97976 10 3.00006C12.0756 3.01691 14 6.49826 14 6.49826H18C18 6.49826 19.9244 3.01691 22 3.00006Z"
    stroke="black"
  />
  <circle cx="11.5" cy="14.5" r="1.5" fill="black" />
  <circle cx="20.5" cy="14.5" r="1.5" fill="black" />
  <path
    d="M13 22.7979C13 22.7979 13.6759 23.0465 14.2857 22.9923C15.2328 22.9081 16 22.4091 16 22.4091M16 22.4091C16 22.4091 16.7672 22.9081 17.7143 22.9923C18.3241 23.0465 19 22.7979 19 22.7979M16 22.4091V20.8541M15 20L16 20.8541M16 20.8541L17 20.0001"
    stroke="black"
    stroke-linecap="round"
  />
  <path
    d="M11 20L5 19"
    stroke="black"
    stroke-width="0.5"
    stroke-linecap="round"
  />
  <path
    d="M11 22L5 23"
    stroke="black"
    stroke-width="0.5"
    stroke-linecap="round"
  />
  <path
    d="M21 20L27 19"
    stroke="black"
    stroke-width="0.5"
    stroke-linecap="round"
  />
  <path
    d="M21 22L27 23"
    stroke="black"
    stroke-width="0.5"
    stroke-linecap="round"
  />
</svg>

<style>
  /* Apply stroke to paths only */
  svg path {
    stroke: gray;
    stroke-width: 0.5;
    stroke-dasharray: 1.5 1; /* Creates dashed lines: 2px dash, 1px gap */
  }
  /* Remove stroke from circles so they render as fully filled */
  svg circle {
    stroke: none;
    fill: gray;
  }

</style>
