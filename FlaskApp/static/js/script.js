Dropzone.options.videoUpload = {
    acceptedFiles: 'video/*', // Only accept video files
    maxFiles: 1,              // Limit to 1 file
    paramName: 'video',
    dictDefaultMessage: "Drop a video file here or click to upload",
    init: function() {
        this.on("maxfilesexceeded", function(file) {
            this.removeAllFiles();
            this.addFile(file);
        });
         this.on("uploadprogress", function(file, progress) {
            console.log("Upload progress:", progress + "%")
            file.previewElement.querySelector("[data-dz-uploadprogress]").style.width = progress + "%";
        });
        this.on("success", function(file, response) {
            if (response) {
                this.removeAllFiles();
                const resultContainer = document.getElementById('result-container')
                const frameContainer = document.getElementById('frame-container')
                frameContainer.innerHTML = '' // Remove all the html elements from frame container initially
                resultContainer.innerHTML = `
                    <h2>
                        Result: <span class="${response.result.toLowerCase()}">${response.result}</span>
                    </h2>
                `

                for (let i = 0; i < response.total_frames_processed; i ++) {
                    let src = '/' + response.frame_directory + '/frame_000' + i + '.png'
                    
                    if (i > 999) {
                        src = '/' + response.frame_directory + '/frame_' + i + '.png'
                    } else if (i > 99) {
                        src = '/' + response.frame_directory + '/frame_0' + i + '.png'
                    } else if (i > 9) {
                        src = '/' + response.frame_directory + '/frame_00' + i + '.png'
                    }

                    const img = document.createElement('img')
                    img.src = src
                    img.classList.add('image-grid')
                    const column = document.createElement('div')
                    column.classList.add('column')
                    column.appendChild(img)
                    frameContainer.appendChild(column)
                }
            }
        });
    }
};