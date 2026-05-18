let chartInstance = null;

// ================= IMAGE PREVIEW =================
function previewImage(event) {

    const preview = document.getElementById("preview");

    preview.src = URL.createObjectURL(event.target.files[0]);
}

// ================= VIDEO PREVIEW =================
function previewVideo(event) {

    const video = document.getElementById("videoPreview");

    video.src = URL.createObjectURL(event.target.files[0]);
}

// ================= UPDATE GRAPH =================
function updateChart(real, fake) {

    const ctx = document.getElementById("chart").getContext("2d");

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {

        type: "bar",

        data: {

            labels: ["REAL", "FAKE"],

            datasets: [{
                label: "Confidence %",
                data: [real, fake],
                backgroundColor: [
                    "#22c55e",
                    "#ef4444"
                ],
                borderRadius: 10
            }]
        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// ================= IMAGE ANALYSIS =================
function analyzeImage() {

    const fileInput = document.getElementById("file");

    if (fileInput.files.length === 0) {

        alert("Please upload an image");

        return;
    }

    const modelType =
        document.getElementById("modelType").value;

    const formData = new FormData();

    formData.append(
        "file",
        fileInput.files[0]
    );

    formData.append(
        "model",
        modelType
    );

    document.getElementById("result").innerHTML =
        " Analyzing Image...";

    fetch("/predict_image", {

        method: "POST",

        body: formData

    })

    .then(response => response.json())

    .then(data => {

        if (data.error) {

            alert(data.error);

            return;
        }

        document.getElementById("result").innerHTML =
            `Prediction: ${data.label}`;

        document.getElementById("confidence").innerHTML =
            `REAL: ${data.real}% | FAKE: ${data.fake}%`;

        updateChart(data.real, data.fake);

        localStorage.setItem(
            "imageResult",
            JSON.stringify(data)
        );
    })

    .catch(error => {

        console.log(error);

        alert("Image analysis failed");
    });
}

// ================= VIDEO ANALYSIS =================
function analyzeVideo() {

    const fileInput =
        document.getElementById("videoFile");

    if (fileInput.files.length === 0) {

        alert("Please upload a video");

        return;
    }

    const modelType =
        document.getElementById("modelType").value;

    const formData = new FormData();

    formData.append(
        "file",
        fileInput.files[0]
    );

    formData.append(
        "model",
        modelType
    );

    document.getElementById("result").innerHTML =
        " Processing Video...";

    fetch("/predict_video", {

        method: "POST",

        body: formData

    })

    .then(response => response.json())

    .then(data => {

        if (data.error) {

            alert(data.error);

            return;
        }

        document.getElementById("result").innerHTML =
            `Prediction: ${data.label}`;

        document.getElementById("confidence").innerHTML =
            `REAL: ${data.real}% | FAKE: ${data.fake}%`;

        updateChart(data.real, data.fake);

        localStorage.setItem(
            "videoResult",
            JSON.stringify(data)
        );
    })

    .catch(error => {

        console.log(error);

        alert("Video analysis failed");
    });
}

// ================= IMAGE RESULT PAGE =================
function loadImageResult() {

    const data = JSON.parse(
        localStorage.getItem("imageResult")
    );

    if (!data) return;

    document.getElementById("info").innerHTML =
        `
        <b>Prediction:</b> ${data.label}
        <br>
        <b>Real:</b> ${data.real}%
        <br>
        <b>Fake:</b> ${data.fake}%
        `;

    document.getElementById("modelUsed").innerHTML =
        `
        <b>Model Used:</b>
        ${data.model === "cnn"
            ? "MobileNetV2"
            : "ResNet18"}
        `;

    document.getElementById("reason").innerHTML =
        data.reason;

    document.getElementById("original").src =
        "/static/uploads/" + data.image;

    // fake heatmap effect
    const heatmap =
        document.getElementById("heatmap");

    heatmap.src =
        "/static/uploads/" + data.image;

    heatmap.style.filter =
        "contrast(180%) hue-rotate(40deg)";
}

// ================= VIDEO RESULT PAGE =================
function loadVideoResult() {

    const data = JSON.parse(
        localStorage.getItem("videoResult")
    );

    if (!data) return;

    document.getElementById("info").innerHTML =
        `
        <b>Prediction:</b> ${data.label}
        <br>
        <b>Real:</b> ${data.real}%
        <br>
        <b>Fake:</b> ${data.fake}%
        `;

    document.getElementById("modelUsed").innerHTML =
        `
        <b>Model Used:</b>
        ${data.model === "cnn"
            ? "MobileNetV2"
            : "ResNet18"}
        `;

    document.getElementById("reason").innerHTML =
        data.reason;

    const framesContainer =
        document.getElementById("frames");

    framesContainer.innerHTML = "";

    data.frames.forEach(frame => {

        const img =
            document.createElement("img");

        img.src =
            "data:image/jpeg;base64," + frame;

        img.className = "frame-image";

        framesContainer.appendChild(img);
    });
}

// ================= RESULT PAGE =================
function goResult() {

    window.location.href = "/image_result";
}

function goVideoResult() {

    window.location.href = "/video_result";
}

// ================= RATING POPUP =================
function goBackWithRating(page) {

    document.getElementById("ratingPopup")
        .style.display = "flex";

    window.nextPage = page;
}

function closePopup() {

    document.getElementById("ratingPopup")
        .style.display = "none";

    window.location.href = window.nextPage;
}

function rate(star) {

    alert("⭐ You rated " + star + " stars");

    closePopup();
}