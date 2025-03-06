<script>
  import { page } from "$app/stores";
  import DisclaimerText from "./DisclaimerText.svelte";
</script>

<footer>
  <div class="footer-text-wrapper">
    {#if $page.route.id === "/cat-pain-detector"}
      <div class="disclaimer-text-wrapper">
        <DisclaimerText />
      </div>
      <br />
    {/if}
    <p class="footer-copyright-text">
      &copy; {new Date().getFullYear()} cat-face-app. All rights reserved.
    </p>
  </div>
</footer>

<style>
  footer {
    width: 100%;
    justify-content: center;
    align-items: center;
    padding: 1rem 20px;
    margin-top: var(--spacing-xlarge);
  }

  .footer-text-wrapper {
    font-size: 12px;
    text-align: center;
  }
</style>
