<script>
  let showGuidelines = false;

  const toggleGuidelines = () => {
    showGuidelines = !showGuidelines;
  };
</script>

<div class="photo-guidelines-wrapper">
  <div>
    <button class="photo-guidelines-button" onclick={toggleGuidelines}>
      Photo guidelines
    </button>
  </div>

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
            alt="Cat not facing the camera"
          />
        </div>
        <div class="example">
          <p>Lying down</p>
          <img src="cat-photo-examples/lying-down.jpeg" alt="Cat lying down" />
        </div>
        <div class="example">
          <p>Too close to the camera</p>
          <img
            src="cat-photo-examples/too-close.jpeg"
            alt="Cat too close to the camera"
          />
        </div>
        <div class="example">
          <p>Unusual posture</p>
          <img
            src="cat-photo-examples/unusual-posture.jpeg"
            alt="Cat with an unusual posture"
          />
        </div>
        <div class="example">
          <p>Mouth open</p>
          <img
            src="cat-photo-examples/mouth-open.jpeg"
            alt="Cat with its mouth open"
          />
        </div>
        <div class="example">
          <p>Squinting</p>
          <img src="cat-photo-examples/squinting.jpeg" alt="Cat squinting" />
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
