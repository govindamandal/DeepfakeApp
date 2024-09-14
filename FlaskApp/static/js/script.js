const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const fileList = document.getElementById('fileList');
    const dropZone = document.querySelector('.drop-zone');

    // Open file dialog on button click
    uploadButton.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', handleFiles);

    // Handle file drag and drop
    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropZone.classList.add('hover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('hover');
    });

    dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropZone.classList.remove('hover');
        if (event.dataTransfer.files.length) {
            fileInput.files = event.dataTransfer.files;
            handleFiles();
        }
    });

    function handleFiles() {
        const files = fileInput.files;
        fileList.innerHTML = '';
        Array.from(files).forEach(file => {
            if (file.type.startsWith('video/')) {
                const fileItem = document.createElement('div');
                fileItem.className = 'text-gray-700 mb-2';
                fileItem.textContent = file.name;
                fileList.appendChild(fileItem);
            } else {
                alert(`${file.name} is not a video file. Please upload a video file.`);
                fileInput.value = ''; // Clear the input
            }
        });
    }