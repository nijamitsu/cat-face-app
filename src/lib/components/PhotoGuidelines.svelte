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
        <p>
          To improve accuracy, take a photo of your cat facing the camera
          directly in a neutral pose, ensuring its entire face, including its
          ears, is visible.
        </p>
      </div>
      <div class="good-photo-example">
        <div class="example">
          <p>A good photo example</p>
          <img
            src="cat-photo-examples/good-photo-example.jpeg"
            alt="Cat not facing the camera"
          />
        </div>
      </div>



      <div class="bad-photo-examples-section">
        <div class="bad-photos-title">
          <p>
            Photos similar to the ones below may not be accurately assessed.
          </p>
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
            <img
              src="cat-photo-examples/lying-down.jpeg"
              alt="Cat lying down"
            />
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
    </div>
  {/if}
</div>

<style>
  .photo-guidelines {
    max-width: 400px;
    font-size: 13px;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-medium);
  }

  .photo-guidelines-button {
    background: none;
    border: none;
    padding: 0;
  }

  .photo-guidelines-button:hover {
    text-decoration: underline;
  }

  .bad-photo-examples-section {
    margin-top: var(--spacing-medium);
  }

  /* Grid layout for examples */
  .bad-photo-examples {
    margin-top: var(--spacing-small);
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
  .photo-guidelines-wrapper img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 10px;
  }

  .example {
    text-align: center;
  }
</style>
