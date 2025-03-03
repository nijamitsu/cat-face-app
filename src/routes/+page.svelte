<script>
  import { onMount, onDestroy } from "svelte";
  import * as tf from "@tensorflow/tfjs";
  import * as cocoSsd from "@tensorflow-models/coco-ssd";

  import Logo from "$lib/elements/Logo.svelte";
  import CatFace from "$lib/elements/CatFace.svelte";
  import PhotoGuidelines from "$lib/components/PhotoGuidelines.svelte";

  let model = $state(null);
  let catDetectionModel = $state(null);
  let fileInput = $state();
  let uploadedImageFile = $state(null);
  let imagePreviewUrl = $state("");
  let catDetectionPreview = $state("");
  let catCroppedPreview = $state("");
  let catPainDiagnosis = $state("");
  let isProcessing = $state(false);
  let hasMounted = $state(false);

  $inspect(catPainDiagnosis);

  $effect(() => {
    if (uploadedImageFile) {
      catPainDiagnosis = "";
    }
  });

  // Load both models when the app is mounted
  onMount(async () => {
    hasMounted = true;
    model = await tf.loadLayersModel("tm-my-image-model/model.json");
    catDetectionModel = await cocoSsd.load();
  });

  onDestroy(() => {
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl);
    }

    if (model) {
      model.dispose();
    }

    if (catDetectionModel) {
      catDetectionModel.dispose();
    }
  });

  // Handle the file input change
  function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
      // Validate file size (e.g. max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        alert("Image too large. Please upload an image under 10MB.");
        return;
      }

      // Validate file type
      if (!["image/jpeg", "image/png", "image/heif"].includes(file.type)) {
        alert("Invalid file type. Please upload a JPEG, PNG or HEIF image.");
        return;
      }

      if (imagePreviewUrl) {
        // Revoke the previous object URL to prevent memory leaks
        URL.revokeObjectURL(imagePreviewUrl);
      }
      uploadedImageFile = file;
      imagePreviewUrl = URL.createObjectURL(file);
      catDetectionPreview = "";
      catCroppedPreview = "";
    }
  }

  // Process the image and make predictions
  async function processImage() {
    if (uploadedImageFile && model && catDetectionModel) {
      isProcessing = true; // Start animation

      const reader = new FileReader();
      reader.onload = async (e) => {
        const img = new Image();
        img.onload = async () => {
          try {
            // Create canvas to detect objects
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // Detect cat
            const predictions = await catDetectionModel.detect(canvas);
            const cat = predictions.find((p) => p.class === "cat");

            if (cat) {
              let [x, y, width, height] = cat.bbox;

              // Expand bounding box to capture whole cat including ears
              const padding = 0.05; // 5% padding
              const expandX = width * padding;
              const expandY = height * padding;

              x = Math.max(0, x - expandX);
              y = Math.max(0, y - expandY);
              width = Math.min(img.width - x, width + 2 * expandX);
              height = Math.min(img.height - y, height + 2 * expandY);

              // Draw detection rectangle on the original canvas
              ctx.strokeStyle = "red";
              ctx.lineWidth = 4;
              ctx.strokeRect(x, y, width, height);
              catDetectionPreview = canvas.toDataURL();

              // Crop the detected cat using the expanded bounding box
              const croppedCanvas = document.createElement("canvas");
              const croppedCtx = croppedCanvas.getContext("2d");
              croppedCanvas.width = width;
              croppedCanvas.height = height;
              croppedCtx.drawImage(
                img,
                x,
                y,
                width,
                height,
                0,
                0,
                width,
                height
              );
              catCroppedPreview = croppedCanvas.toDataURL();

              // Convert cropped image to tensor
              const tensor = tf.tidy(() => {
                return tf.browser
                  .fromPixels(croppedCanvas)
                  .resizeNearestNeighbor([224, 224])
                  .toFloat()
                  .div(tf.scalar(255))
                  .expandDims(0);
              });

              // Make prediction using your existing model
              const prediction = model.predict(tensor);
              const predictionArray = await prediction.data();
              prediction.dispose();

              // Dispose of the tensor since it's no longer needed
              tensor.dispose();

              /* console.log(predictionArray); */

              // Calculate percentage for each class
              const absentPercentage = (predictionArray[0] * 100).toFixed(2);
              const moderatelyPresentPercentage = (
                predictionArray[1] * 100
              ).toFixed(2);
              const markedlyPresentPercentage = (
                predictionArray[2] * 100
              ).toFixed(2);

              // Define class labels
              const classLabels = [
                "absent",
                "moderatelyPresent",
                "markedlyPresent",
              ];

              // Convert percentages back to decimal probabilities
              const probabilities = [
                parseFloat(absentPercentage) / 100,
                parseFloat(moderatelyPresentPercentage) / 100,
                parseFloat(markedlyPresentPercentage) / 100,
              ];

              // Find the two highest probabilities
              const sortedIndices = [...probabilities.keys()].sort(
                (a, b) => probabilities[b] - probabilities[a]
              );
              const maxIndex = sortedIndices[0];
              const secondMaxIndex = sortedIndices[1];

              const maxProbability = probabilities[maxIndex];
              const secondMaxProbability = probabilities[secondMaxIndex];
              const predictedClass = classLabels[maxIndex];

              // Set thresholds
              const CONFIDENCE_THRESHOLD = 0.7;
              const CLOSE_DIFFERENCE_THRESHOLD = 0.05; // 5% difference

              // Determine the final message
              let finalMessage;
              if (
                maxProbability >= CONFIDENCE_THRESHOLD &&
                maxProbability - secondMaxProbability >=
                  CLOSE_DIFFERENCE_THRESHOLD
              ) {
                switch (predictedClass) {
                  case "absent":
                    finalMessage = "😻 Your cat seems comfortable!";
                    break;
                  case "moderatelyPresent":
                    finalMessage =
                      "😼 Your cat appears to be in mild pain. If this result is consistent across multiple photos, consider consulting your vet.";
                    break;
                  case "markedlyPresent":
                    finalMessage =
                      "😿 Your cat appears to be in significant pain. If this result is consistent across multiple photos, consider consulting your vet.";
                    break;
                }
              } else {
                finalMessage = "🤔 We're not sure. Try a different photo.";
              }

              // Store the result for UI display
              catPainDiagnosis = {
                absent: absentPercentage,
                moderatelyPresent: moderatelyPresentPercentage,
                markedlyPresent: markedlyPresentPercentage,
                confidence: (maxProbability * 100).toFixed(2),
                message: finalMessage,
              };
            } else {
              catPainDiagnosis = { message: "No cat detected in the photo" };
            }
          } finally {
            isProcessing = false; // End animation regardless of success/failure
          }
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(uploadedImageFile);
    }
  }
</script>

<section class="main-section">
  <div class="main-container">
    <div class="logo-wrapper"><Logo size={32} /></div>
    <div class="welcome-text">
      <h1>
        Analyze your cat's <br /> comfort level using
        <span class="ai-text">AI</span>
      </h1>
    </div>

    <!-- Hidden file input -->
    <input
      type="file"
      accept="image/png, image/jpeg, image/heif"
      onchange={handleFileUpload}
      bind:this={fileInput}
      style="display: none;"
    />

    <!-- Droppable & clickable upload area with mutual wrapping div for dynamic borders -->
    <div class="image-upload-wrapper">
      <div class="upload-area {imagePreviewUrl ? 'uploaded' : 'not-uploaded'}">
        {#if imagePreviewUrl}
          <button
            class="image-upload-preview-button"
            onclick={() => fileInput.click()}
          >
            <img
              class={`uploaded-image${isProcessing ? " processing" : ""}`}
              src={catDetectionPreview || imagePreviewUrl}
              alt={catDetectionPreview
                ? "Cat detection preview"
                : "Upload preview"}
            />
          </button>
        {:else}
          <button
            class="image-upload-dragdrop-area-button"
            onclick={() => fileInput.click()}
            ondragover={(e) => e.preventDefault()}
            ondrop={handleFileUpload}
            tabindex="0"
          >
            {#if hasMounted}
              <div class="cat-face-svg-wrapper">
                <CatFace />
              </div>
              {#if navigator.maxTouchPoints > 0}
                Tap to take a photo or upload one
              {:else}
                Click or drop your photo here
              {/if}
            {/if}
          </button>
        {/if}
      </div>
      <button
        class="analyze-button"
        onclick={processImage}
        disabled={!imagePreviewUrl || catPainDiagnosis.message || isProcessing}
      >
        {isProcessing ? "Processing..." : "Analyze"}
      </button>
    </div>

    <div class="catPainDiagnosis-message-div">
      {#if catPainDiagnosis.message}
        <p>{catPainDiagnosis.message}</p>
      {/if}
    </div>

    <PhotoGuidelines />
  </div>
</section>

<style>
  .main-section {
    width: 100%;
    max-width: 800px;
    margin: 0 auto;
    flex: 1;
    padding: 0 var(--spacing-large);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .main-container {
    width: 100%;
    max-width: 400px;
    margin-top: var(--spacing-large);
  }

  .logo-wrapper {
    display: flex;
    justify-content: center;
  }

  .welcome-text {
    margin-top: var(--spacing-large);
    text-align: center;
  }

  .ai-text {
    position: relative;
  }

  .ai-text::after {
    content: "beta";
    position: absolute;
    top: -5px;
    right: -30px;
    font-size: 12px;
  }

  .image-upload-wrapper {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  /* Mutual wrapping div styles */

  .upload-area {
    width: 100%;
    aspect-ratio: 1;
    max-height: 400px;
    display: flex;
    border: 2px grey;
    border-radius: 10px;
  }

  .upload-area:hover {
    border-color: var(--color-primary);
    box-shadow: 0 0 10px 2px #c0c0c033;
  }

  /* Define border style based on state */
  .upload-area.not-uploaded {
    border-style: dashed;
  }
  .upload-area.uploaded {
    border-style: solid;
    border-color: var(--color-primary);
  }

  .upload-area.uploaded:hover {
    border-style: dashed;
  }

  .upload-area button {
    width: 100%;
    border: none;
    padding: 0;
    background: none;
    cursor: pointer;
    color: grey;
  }

  .upload-area button:hover {
    color: var(--color-primary);
  }

  .upload-area:hover :global(svg path) {
    stroke: var(--color-primary);
  }

  .upload-area:hover :global(svg circle) {
    fill: var(--color-primary);
  }

  .uploaded-image {
    width: 100%;
    height: 100%;
    object-fit: contain;
    border-radius: 10px;
  }

  .cat-face-svg-wrapper {
    width: 150px;
    height: 150px;
    margin: 0 auto;
  }

  .analyze-button {
    background: #303134;
    border: none;
    border-radius: 10px;
    padding: 10px;
    cursor: pointer;
  }

  .analyze-button:not(:disabled):hover {
    box-shadow: 0 0 10px 2px #c0c0c033;
  }

  .analyze-button:disabled {
    background: none;
    cursor: default;
    color: gray;
    border: 2px solid gray;
  }

  /* Add these styles */
  /* Negative effect that alternates with normal appearance */
  .uploaded-image.processing {
    animation: negativeEffect 3s infinite alternate;
  }

  @keyframes negativeEffect {
    0% {
      filter: none; /* Normal image */
    }
    50% {
      filter: invert(0.8) contrast(1.2) brightness(0.9); /* Full negative effect */
    }
    100% {
      filter: none; /* Back to normal */
    }
  }

  .catPainDiagnosis-message-div {
    margin-top: var(--spacing-large);
    margin-bottom: var(--spacing-large);
  }
</style>
