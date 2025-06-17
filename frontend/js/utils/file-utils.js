// File Utility Functions
class FileUtils {
    // File validation
    static isValidAudioFile(file) {
        if (!file) return false;

        const validExtensions = AppConfig.AUDIO.SUPPORTED_FORMATS;
        const validMimeTypes = AppConfig.AUDIO.MIME_TYPES;

        const fileName = file.name.toLowerCase();
        const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
        const hasValidMimeType = validMimeTypes.includes(file.type);

        return hasValidExtension || hasValidMimeType;
    }

    static getFileValidationError(file) {
        if (!file) return 'No file selected';

        if (!this.isValidAudioFile(file)) {
            return `Please select a valid audio file (${AppConfig.AUDIO.SUPPORTED_FORMATS.join(', ')})`;
        }

        if (file.size > AppConfig.UI.MAX_FILE_SIZE) {
            return `File too large. Maximum size is ${this.formatFileSize(AppConfig.UI.MAX_FILE_SIZE)}`;
        }

        return null;
    }

    // File size formatting
    static formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';

        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));

        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // File type detection
    static getFileType(file) {
        if (file.type) return file.type;

        const extension = file.name.split('.').pop().toLowerCase();
        const mimeTypes = {
            'wav': 'audio/wav',
            'flac': 'audio/flac',
            'mp3': 'audio/mpeg',
            'ogg': 'audio/ogg',
            'webm': 'audio/webm'
        };

        return mimeTypes[extension] || 'application/octet-stream';
    }

    // File reading utilities
    static readFileAsArrayBuffer(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(file);
        });
    }

    static readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsText(file);
        });
    }

    static readFileAsDataURL(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });
    }

    // File download utilities
    static downloadFile(content, filename, mimeType = 'text/plain') {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        URL.revokeObjectURL(url);
    }

    static downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.style.display = 'none';

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        URL.revokeObjectURL(url);
    }

    // Audio file duration detection
    static getAudioDuration(file) {
        return new Promise((resolve, reject) => {
            const audio = new Audio();
            const url = URL.createObjectURL(file);

            audio.addEventListener('loadedmetadata', () => {
                URL.revokeObjectURL(url);
                resolve(audio.duration);
            });

            audio.addEventListener('error', () => {
                URL.revokeObjectURL(url);
                reject(new Error('Could not load audio file'));
            });

            audio.src = url;
        });
    }

    // Drag and drop utilities
    static setupDragAndDrop(element, callbacks = {}) {
        const {
            onDragEnter = () => {},
            onDragLeave = () => {},
            onDragOver = () => {},
            onDrop = () => {},
            onFiles = () => {}
        } = callbacks;

        let dragCounter = 0;

        element.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dragCounter++;
            if (dragCounter === 1) {
                onDragEnter(e);
            }
        });

        element.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter === 0) {
                onDragLeave(e);
            }
        });

        element.addEventListener('dragover', (e) => {
            e.preventDefault();
            onDragOver(e);
        });

        element.addEventListener('drop', (e) => {
            e.preventDefault();
            dragCounter = 0;

            const files = Array.from(e.dataTransfer.files);
            onDrop(e, files);
            onFiles(files);
        });
    }

    // File input utilities
    static createFileInput(options = {}) {
        const {
            accept = '*/*',
            multiple = false,
            onChange = () => {}
        } = options;

        const input = document.createElement('input');
        input.type = 'file';
        input.accept = accept;
        input.multiple = multiple;
        input.style.display = 'none';

        input.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            onChange(files);
        });

        return input;
    }

    // File compression utilities
    static async compressImage(file, quality = 0.8, maxWidth = 1920, maxHeight = 1080) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                // Calculate new dimensions
                let { width, height } = img;

                if (width > maxWidth || height > maxHeight) {
                    const ratio = Math.min(maxWidth / width, maxHeight / height);
                    width *= ratio;
                    height *= ratio;
                }

                // Set canvas size and draw image
                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(img, 0, 0, width, height);

                // Convert to blob
                canvas.toBlob(resolve, 'image/jpeg', quality);
            };

            img.src = URL.createObjectURL(file);
        });
    }

    // Batch file processing
    static async processFilesInBatches(files, processor, batchSize = 5) {
        const results = [];

        for (let i = 0; i < files.length; i += batchSize) {
            const batch = files.slice(i, i + batchSize);
            const batchPromises = batch.map(processor);
            const batchResults = await Promise.allSettled(batchPromises);
            results.push(...batchResults);
        }

        return results;
    }

    // File comparison utilities
    static filesAreEqual(file1, file2) {
        return file1.name === file2.name &&
               file1.size === file2.size &&
               file1.type === file2.type &&
               file1.lastModified === file2.lastModified;
    }

    static deduplicateFiles(files) {
        const seen = new Set();
        const unique = [];

        for (const file of files) {
            const key = `${file.name}-${file.size}-${file.lastModified}`;
            if (!seen.has(key)) {
                seen.add(key);
                unique.push(file);
            }
        }

        return unique;
    }

    // File metadata extraction
    static getFileMetadata(file) {
        return {
            name: file.name,
            size: file.size,
            type: file.type,
            lastModified: file.lastModified,
            lastModifiedDate: new Date(file.lastModified),
            extension: file.name.split('.').pop().toLowerCase()
        };
    }

    // URL utilities
    static createObjectURL(file) {
        return URL.createObjectURL(file);
    }

    static revokeObjectURL(url) {
        URL.revokeObjectURL(url);
    }

    // File system access (if supported)
    static async saveFile(content, filename, options = {}) {
        if ('showSaveFilePicker' in window) {
            try {
                const fileHandle = await window.showSaveFilePicker({
                    suggestedName: filename,
                    ...options
                });

                const writable = await fileHandle.createWritable();
                await writable.write(content);
                await writable.close();

                return true;
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Error saving file:', error);
                }
                return false;
            }
        } else {
            // Fallback to download
            this.downloadFile(content, filename, options.types?.[0]?.accept?.['*/*'] || 'text/plain');
            return true;
        }
    }

    static async openFile(options = {}) {
        if ('showOpenFilePicker' in window) {
            try {
                const [fileHandle] = await window.showOpenFilePicker(options);
                const file = await fileHandle.getFile();
                return file;
            } catch (error) {
                if (error.name !== 'AbortError') {
                    console.error('Error opening file:', error);
                }
                return null;
            }
        } else {
            // Fallback to file input
            return new Promise((resolve) => {
                const input = this.createFileInput({
                    accept: options.types?.map(type =>
                        Object.keys(type.accept).join(',')
                    ).join(',') || '*/*',
                    onChange: (files) => resolve(files[0] || null)
                });

                document.body.appendChild(input);
                input.click();
                document.body.removeChild(input);
            });
        }
    }

    // Validation utilities
    static validateFileList(files, rules = {}) {
        const {
            maxFiles = Infinity,
            maxSize = Infinity,
            minSize = 0,
            allowedTypes = [],
            requiredTypes = []
        } = rules;

        const errors = [];

        if (files.length > maxFiles) {
            errors.push(`Too many files. Maximum allowed: ${maxFiles}`);
        }

        for (const file of files) {
            if (file.size > maxSize) {
                errors.push(`File "${file.name}" is too large. Maximum size: ${this.formatFileSize(maxSize)}`);
            }

            if (file.size < minSize) {
                errors.push(`File "${file.name}" is too small. Minimum size: ${this.formatFileSize(minSize)}`);
            }

            if (allowedTypes.length > 0 && !allowedTypes.includes(file.type)) {
                errors.push(`File "${file.name}" has unsupported type: ${file.type}`);
            }
        }

        if (requiredTypes.length > 0) {
            const fileTypes = files.map(f => f.type);
            for (const requiredType of requiredTypes) {
                if (!fileTypes.includes(requiredType)) {
                    errors.push(`Missing required file type: ${requiredType}`);
                }
            }
        }

        return {
            valid: errors.length === 0,
            errors
        };
    }
}
