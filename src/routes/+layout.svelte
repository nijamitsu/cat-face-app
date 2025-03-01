<script>
  import Footer from "$lib/components/Footer.svelte";
  import "../app.css";

  /**
   * @typedef {Object} Props
   * @property {import('svelte').Snippet} [children]
   */

  /** @type {Props} */
  let { children } = $props();
</script>

<main>
  {@render children?.()}
  <Footer />
</main>

<style>
  main {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
</style>
