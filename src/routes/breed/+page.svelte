<script>
  import { onMount, onDestroy } from "svelte";
  import * as tf from "@tensorflow/tfjs";
  import * as cocoSsd from "@tensorflow-models/coco-ssd";

  import Logo from "$lib/elements/Logo.svelte";
  import CatFace from "$lib/elements/CatFace.svelte";
  import PhotoGuidelines from "$lib/components/PhotoGuidelines.svelte";

  let model;
  let catDetector;
  let fileInput = $state();
  let uploadedImageFile = $state(null);
  let imagePreviewUrl = $state("");
  let isProcessing = $state(false);
  let predictionMessage = $state("");
  let hasMounted = $state(false);
  let catDetectionPreview = $state("");
  let catCroppedPreview = $state("");

  const classNames = [
    "Abyssinian", // done
    "Birman", 
    "American Shorthair", // getting confused with ragdoll
    "Bengal", // done
    "Ragdoll", // done
    "Bombay", // done
    "British Shorthair", // done
    "Egyptian Mau", // done
    "Maine Coon", // done
    "Persian", // done
    "American Bobtail",
    "Russian Blue", // done
    "Siamese", // done
    "Sphynx", // done
    "Tuxedo", // done
  ];

  $effect(() => {
    if (uploadedImageFile) {
      predictionMessage = "";
    }
  });

  // Load the models only after the component is mounted on the client-side
  onMount(async () => {
    hasMounted = true;
    model = await tf.loadGraphModel("cat-breed-identify-tfjs_model/model.json");
    catDetector = await cocoSsd.load();
  });

async function processImage() {
  if (!uploadedImageFile || !model || !catDetector) {
    console.error("Image or models not loaded!");
    return;
  }

  isProcessing = true;
  await new Promise((resolve) => setTimeout(resolve, 100));

  try {
    // Create an image element to use with TensorFlow and coco-ssd
    const img = new Image();
    img.src = imagePreviewUrl;

    await new Promise((resolve) => {
      img.onload = resolve;
    });

    // Create a canvas to draw the image for detection
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);

    // Use coco-ssd to detect cat in the image
    const predictions = await catDetector.detect(canvas);
    const catPrediction = predictions.find(pred => pred.class === "cat");

    if (!catPrediction) {
      predictionMessage = "No cat detected in the image. Please try another photo.";
      return;
    }

    // Extract and adjust the bounding box with 5% padding
    const [x, y, width, height] = catPrediction.bbox;
    const padX = width * 0.05;
    const padY = height * 0.05;
    const newX = Math.max(0, x - padX);
    const newY = Math.max(0, y - padY);
    const newWidth = Math.min(img.width - newX, width + padX * 2);
    const newHeight = Math.min(img.height - newY, height + padY * 2);

    // Draw the detection box on the original canvas
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(newX, newY, newWidth, newHeight);
    
    // Store the detection preview (canvas with overlay)
    catDetectionPreview = canvas.toDataURL();

    // Create a temporary canvas to crop the detected cat area
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = newWidth;
    tempCanvas.height = newHeight;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(canvas, newX, newY, newWidth, newHeight, 0, 0, newWidth, newHeight);

    // Store the cropped image preview
    catCroppedPreview = tempCanvas.toDataURL();

    // Process the cropped image with the breed model
    const result = await predictCatBreed(tempCanvas);

    // Update the UI with the results
    predictionMessage = `This appears to be a ${result.predictedClassName} cat with ${Math.round(
      result.predictedClassProbability * 100
    )}% confidence.`;
  } catch (error) {
    console.error("Error processing image:", error);
    predictionMessage = "Error analyzing the image. Please try again.";
  } finally {
    isProcessing = false;
  }
}

  // Helper function to process the image with the model
  async function predictCatBreed(image) {
    if (!model) {
      console.error("Model is not loaded yet!");
      throw new Error("Model not loaded");
    }

    const imgTensor = tf.browser
      .fromPixels(image)
      .resizeNearestNeighbor([224, 224]) // Resize to model input size
      .toFloat()
      .expandDims();

    const prediction = model.predict(imgTensor);
    const result = await prediction.data();

    // Find the index of the highest value
    const predictedClassIndex = result.indexOf(Math.max(...result));
    const predictedClassName = classNames[predictedClassIndex];
    const predictedClassProbability = result[predictedClassIndex];

    // Clean up tensors to prevent memory leaks
    imgTensor.dispose();
    prediction.dispose();

    return { predictedClassName, predictedClassProbability };
  }

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
        URL.revokeObjectURL(imagePreviewUrl);
      }
      uploadedImageFile = file;
      imagePreviewUrl = URL.createObjectURL(file);
      catDetectionPreview = "";
      catCroppedPreview = "";
    }
  }
</script>

<section class="main-section">
  <div class="main-container">
    <div class="logo-wrapper"><Logo size={32} /></div>
    <div class="welcome-text">
      <h1>
        Identify cat breeds<br />with our AI
        <span class="ai-text">model</span>
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
        disabled={!imagePreviewUrl || predictionMessage || isProcessing}
      >
        {isProcessing ? "Processing..." : "Analyze"}
      </button>
    </div>

    <div class="catPainDiagnosis-message-div">
      {#if predictionMessage}
        <p>{predictionMessage}</p>
      {/if}
    </div>
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