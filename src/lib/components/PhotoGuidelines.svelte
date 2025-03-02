<script>
  let showGuidelines = false;

  const toggleGuidelines = () => {
    showGuidelines = !showGuidelines;
  };
</script>

<div class="photo-guidelines-wrapper">
  <button class="photo-guidelines-button" onclick={toggleGuidelines}>
    See photo guidelines
  </button>

  {#if showGuidelines}
    <div class="photo-guidelines">
      <div class="description">
        <p>Photos similar to the ones below may not be accurately assessed.</p>
      </div>
      <div class="bad-photo-examples">
        <div class="example">
          <p>Not facing the camera</p>
          <img
            src="cat-photo-examples/not-facing-camera.jpeg"
            alt="Not directly facing the camera"
          />
        </div>
        <div class="example">
          <p>Lying down</p>
          <img src="cat-photo-examples/lying-down.jpeg" alt="Lying down" />
        </div>
        <div class="example">
          <p>Too close to the camera</p>
          <img
            src="cat-photo-examples/too-close.jpeg"
            alt="Too close to the camera"
          />
        </div>
        <div class="example">
          <p>Unusual posture</p>
          <img
            src="cat-photo-examples/unusual-posture.jpeg"
            alt="Unusual posture"
          />
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .photo-guidelines {
    max-width: 400px;
    font-size: 13px;
  }

  .photo-guidelines-button {
    margin-bottom: 0.5rem;
    background: none;
    border: none;
    padding: 0;
  }

  .photo-guidelines-button:hover {
    text-decoration: underline;
  }

  .description p {
    margin-bottom: 1rem;
  }

  /* Grid layout for examples */
  .bad-photo-examples {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: var(--spacing-medium);
  }

  /* Each example is centered using flexbox */
  .bad-photo-examples .example {
    display: flex;
    flex-direction: column;
    align-items: center;
    border-radius: 10px;
  }

  /* Images responsive styling */
  .bad-photo-examples img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 10px;
  }
</style>
