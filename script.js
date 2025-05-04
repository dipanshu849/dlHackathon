document.addEventListener('DOMContentLoaded', () => {
    // Get references to the HTML elements
    const imageUpload = document.getElementById('imageUpload');
    const originalImage = document.getElementById('originalImage');
    const segmentedImage = document.getElementById('segmentedImage');
    const originalPlaceholder = document.getElementById('originalPlaceholder');
    const segmentedPlaceholder = document.getElementById('segmentedPlaceholder');
    const loadingMessage = document.getElementById('loadingMessage');

    // Add an event listener to the file input
    imageUpload.addEventListener('change', handleImageUpload);

    /**
     * Handles the file upload event.
     * Reads the selected file, displays the original image, and
     * sends it to the server for segmentation.
     * @param {Event} event The change event from the file input.
     */
    function handleImageUpload(event) {
        const file = event.target.files[0]; // Get the first selected file

        if (file) {
            const reader = new FileReader(); // Create a FileReader to read the file

            reader.onload = function(e) {
                // Display the original image immediately
                originalImage.src = e.target.result;
                originalImage.classList.remove('hidden'); // Show the image
                originalPlaceholder.classList.add('hidden'); // Hide the placeholder

                // Reset the segmented image area
                segmentedImage.classList.add('hidden'); // Hide previous segmented image
                segmentedImage.src = '#'; // Clear the source
                segmentedPlaceholder.classList.remove('hidden'); // Show the segmented placeholder

                // Show loading message
                loadingMessage.textContent = 'Processing image...';
                loadingMessage.classList.remove('hidden');

                // --- Server Communication ---
                // Send the image file to your Flask server for segmentation
                const formData = new FormData();
                formData.append('eye_image', file); // 'eye_image' must match the key used in Flask's request.files

                // *** UPDATED URL TO USE PORT 5001 ***
                // If running Flask on your local machine on port 5001:
                // 'http://127.0.0.1:5001/segment_image' or 'http://localhost:5001/segment_image'
                fetch('http://127.0.0.0:5001/segment_image', { // Using 0.0.0.0 might be necessary if accessing from another machine
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    if (!response.ok) {
                        // If the server responded with an error status code (e.g., 400, 500)
                        // Try to read the error message from the response body
                        return response.json().then(err => {
                            throw new Error('Server error: ' + (err.error || response.statusText));
                        }).catch(() => {
                             // If response is not JSON, just throw a generic error
                             throw new Error('Server error: ' + response.statusText);
                        });
                    }
                    // If the response is OK, assume it's the image data (Blob)
                    return response.blob();
                })
                .then(segmentedBlob => {
                    // Assuming the server sends back the segmented image as a Blob
                    // Create a URL for the Blob and set it as the source for the segmented image
                    const segmentedImageUrl = URL.createObjectURL(segmentedBlob);
                    segmentedImage.src = segmentedImageUrl; // Display the segmented image
                    segmentedImage.classList.remove('hidden');
                    segmentedPlaceholder.classList.add('hidden');
                    loadingMessage.classList.add('hidden'); // Hide loading message

                    // Clean up the object URL after the image has loaded to free up memory
                    segmentedImage.onload = () => {
                        URL.revokeObjectURL(segmentedImageUrl);
                    };
                })
                .catch(error => {
                    // Handle any errors during the fetch request or server processing
                    console.error('Error processing image:', error);
                    loadingMessage.textContent = 'Error processing image: ' + error.message; // Display error message to the user
                    loadingMessage.classList.remove('hidden');
                    // Optionally, hide the segmented image area again on error
                    segmentedImage.classList.add('hidden');
                    segmentedPlaceholder.classList.remove('hidden');
                });
                // --- End Server Communication ---
            }

            // Read the file as a data URL to display the original image immediately in the browser
            reader.readAsDataURL(file);
        }
    }
});
