Dropzone.options.videoUpload = {
    acceptedFiles: 'video/*', // Only accept video files
    maxFiles: 1,              // Limit to 1 file
    dictDefaultMessage: "Drop a video file here or click to upload",
    init: function() {
        this.on("maxfilesexceeded", function(file) {
            this.removeAllFiles(); // Remove any existing files
            this.addFile(file);    // Add the new file
        });
    }
};