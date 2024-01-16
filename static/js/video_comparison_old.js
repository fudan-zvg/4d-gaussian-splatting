// Written by Dor Verbin, October 2021
// This is based on: http://thenewcode.com/364/Interactive-Before-and-After-Video-Comparison-in-HTML5-Canvas
// With additional modifications based on: https://jsfiddle.net/7sk5k4gp/13/

function playVids(videoId) {
    var videoMerge = document.getElementById(videoId + "Merge");
    var vid = document.getElementById(videoId);

    var position = 0.5;
    var vidWidth = vid.videoWidth / 2;
    var vidHeight = vid.videoHeight;

    var mergeContext = videoMerge.getContext("2d");

    if (vid.readyState > 3) {
        vid.play();

        var isDragging = false; // Variable to track if dragging is in progress

        // Function to start dragging
        function startDrag(e) {
            isDragging = true;
            handleDrag(e);
        }

        // Function to end dragging
        function endDrag() {
            isDragging = false;
        }

        // Function to handle dragging
        function handleDrag(e) {
            if (isDragging) {
                // Normalize to [0, 1]
                var bcr = videoMerge.getBoundingClientRect();
                position = ((e.clientX - bcr.x) / bcr.width).clamp(0.0, 1.0);
            }
        }

        // Event listeners for mouse events
        videoMerge.addEventListener("mousedown", startDrag, false);
        window.addEventListener("mouseup", endDrag, false);
        window.addEventListener("mousemove", handleDrag, false);

        var lastDrawTime = 0;

        function drawLoop() {

            if (timestamp - lastDrawTime < 100) { // Limit to around 60 fps
                requestAnimationFrame(drawLoop);
                return;
            }
            lastDrawTime = timestamp;

            mergeContext.clearRect(0, 0, videoMerge.width, videoMerge.height);

            mergeContext.drawImage(vid, 0, 0, vidWidth, vidHeight, 0, 0, vidWidth, vidHeight);
            var colStart = (vidWidth * position).clamp(0.0, vidWidth);
            var colWidth = (vidWidth - (vidWidth * position)).clamp(0.0, vidWidth);
            mergeContext.drawImage(vid, colStart + vidWidth, 0, colWidth, vidHeight, colStart, 0, colWidth, vidHeight);

            // requestAnimationFrame(drawLoop);

            var arrowLength = 0.09 * vidHeight;
            var arrowheadWidth = 0.025 * vidHeight;
            var arrowheadLength = 0.04 * vidHeight;
            var arrowPosY = vidHeight / 2;
            var arrowWidth = 0.007 * vidHeight;
            var currX = vidWidth * position;

            // Draw circle
            mergeContext.arc(currX, arrowPosY, arrowLength * 0.7, 0, Math.PI * 2, false);
            mergeContext.fillStyle = "#FFD79340";
            mergeContext.fill()
            //mergeContext.strokeStyle = "#444444";
            //mergeContext.stroke()

            // Draw border
            mergeContext.beginPath();
            mergeContext.moveTo(vidWidth * position, 0);
            mergeContext.lineTo(vidWidth * position, vidHeight);
            mergeContext.closePath()
            mergeContext.strokeStyle = "#444444";
            mergeContext.lineWidth = 5;
            mergeContext.stroke();

            // Draw arrow
            mergeContext.beginPath();
            mergeContext.moveTo(currX, arrowPosY - arrowWidth / 2);

            // Move right until meeting arrow head
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowWidth / 2);

            // Draw right arrow head
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2, arrowPosY);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
            mergeContext.lineTo(currX + arrowLength / 2 - arrowheadLength / 2, arrowPosY + arrowWidth / 2);

            // Go back to the left until meeting left arrow head
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowWidth / 2);

            // Draw left arrow head
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY + arrowheadWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2, arrowPosY);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowheadWidth / 2);
            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY);

            mergeContext.lineTo(currX - arrowLength / 2 + arrowheadLength / 2, arrowPosY - arrowWidth / 2);
            mergeContext.lineTo(currX, arrowPosY - arrowWidth / 2);

            mergeContext.closePath();

            mergeContext.fillStyle = "#444444";
            mergeContext.fill();


        }
        // requestAnimationFrame(drawThrottled);
        requestAnimationFrame(drawLoop);
    }
}

Number.prototype.clamp = function (min, max) {
    return Math.min(Math.max(this, min), max);
};


function resizeAndPlay(element) {
    var cv = document.getElementById(element.id + "Merge");
    cv.width = element.videoWidth / 2;
    cv.height = element.videoHeight;
    element.play();
    element.style.height = "0px";  // Hide video without stopping it

    playVids(element.id);
}
