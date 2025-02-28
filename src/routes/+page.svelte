<script>
  import { onMount } from "svelte";
  import * as tf from "@tensorflow/tfjs";
  import * as cocoSsd from "@tensorflow-models/coco-ssd";
  import Logo from "$lib/elements/Logo.svelte";

  let model = $state(null);
  let catDetectionModel = $state(null);
  let uploadedImageFile = $state(null);
  let catPainDiagnosis = $state("");
  let imagePreviewUrl = $state("");
  let catDetectionPreview = $state("");
  let catCroppedPreview = $state("");
  let fileInput = $state();

  let isProcessing = $state(false);

  $effect(() => {
    if (uploadedImageFile) {
      catPainDiagnosis = "";
    }
  });

  // Load both models when the app is mounted
  onMount(async () => {
    model = await tf.loadLayersModel("tm-my-image-model/model.json");
    catDetectionModel = await cocoSsd.load();
    console.log("Model loaded");
  });

  // Handle the file input change
  function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
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
              const tensor = tf.browser
                .fromPixels(croppedCanvas)
                .resizeNearestNeighbor([224, 224])
                .toFloat()
                .div(tf.scalar(255))
                .expandDims(0);

              // Make prediction using your existing model
              const prediction = model.predict(tensor);
              const predictionArray = await prediction.data();

              console.log(predictionArray);

              // Calculate percentage for each class
              const absentPercentage = (predictionArray[0] * 100).toFixed(2);
              const moderatelyPresentPercentage = (
                predictionArray[1] * 100
              ).toFixed(2);
              const markedlyPresentPercentage = (
                predictionArray[2] * 100
              ).toFixed(2);

              // Determine the final result
              let finalMessage =
                "😺 Your cat does not appear to be in pain. To be certain, try uploading a few more photos.";
              if (markedlyPresentPercentage >= 70) {
                finalMessage =
                  "😿 Your cat appear to be in significant pain. If you consistently get this result with multiple photos, consider consulting your vet.";
              } else if (moderatelyPresentPercentage >= 30) {
                finalMessage =
                  "🐱 Your cat appear to be in mild pain. If you consistently get this result with multiple photos, consider consulting your vet.";
              }

              // Store the result for UI display
              catPainDiagnosis = {
                absent: absentPercentage,
                moderatelyPresent: moderatelyPresentPercentage,
                markedlyPresent: markedlyPresentPercentage,
                message: finalMessage,
              };
            } else {
              catPainDiagnosis = { message: "No cat detected in the photo!" };
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
    <div class="welcome-text"><h1>Is your cat in pain?</h1></div>

    <!-- Hidden file input -->
    <input
      type="file"
      accept="image/png, image/jpeg, image/heif"
      onchange={handleFileUpload}
      bind:this={fileInput}
      style="display: none;"
    />

    <!-- Droppable & clickable upload area -->
    <div class="image-upload-wrapper">
      {#if imagePreviewUrl}
        <button
          class="image-upload-preview-button"
          onclick={() => fileInput.click()}
        >
          <img
            class="preview-image {isProcessing ? 'processing' : ''}"
            src={imagePreviewUrl}
            alt="Upload preview"
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
          <p>Click or drop your image here</p>
        </button>
      {/if}
      <button
        class="analyze-button"
        onclick={processImage}
        disabled={!imagePreviewUrl || catPainDiagnosis.message || isProcessing}
      >
        {isProcessing ? "Processing..." : "Analyze"}
      </button>
    </div>

    <div>
      <!-- COCO-SSD Cat Detection Preview -->
      {#if catDetectionPreview}
        <div>
          <img
            class="image-cat-detection-preview"
            src={catDetectionPreview}
            alt="Cat detection preview"
          />
        </div>
      {/if}
    </div>

    <div>
      {#if catPainDiagnosis.message}
        <p>{catPainDiagnosis.message}</p>
      {/if}
    </div>
  </div>
</section>

<style>
  .main-section {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .main-container {
    max-width: 300px;
    margin-top: 30px;
  }

  .logo-wrapper {
    display: flex;
    justify-content: center;
  }

  .welcome-text {
    text-align: center;
    margin-top: 20px;
  }

  .image-upload-wrapper {
    max-width: 300px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .image-upload-dragdrop-area-button {
    background: none;
    width: 300px;
    height: 300px;
    border: 2px dashed grey;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border-radius: 10px;
  }

  .image-upload-dragdrop-area-button:hover {
    border: 2px solid grey;
  }

  .preview-image {
    width: 300px;
    height: 300px;
    object-fit: contain;
    cursor: pointer;
    border-radius: 10px;
    border: 2px solid grey;
  }

  .preview-image:hover {
    filter: brightness(1.05);
    box-shadow: 0 0 10px 2px #c0c0c033;
    border: 2px dashed grey;
  }

  .image-upload-preview-button {
    display: inline-block;
    padding: 0;
    border: none;
    background: none;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .analyze-button {
    background: #303134;
    border: none;
    border-radius: 10px;
    width: 300px;
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

  .image-cat-detection-preview {
    width: 300px;
    height: 300px;
    object-fit: contain;
    border-radius: 10px;
    border: 2px solid transparent;
  }

  .analyze-button:disabled {
    background: none;
    cursor: default;
    color: gray;
  }

  /* Add these styles */
  /* Negative effect that alternates with normal appearance */
  .preview-image.processing {
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
</style>
