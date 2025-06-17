// Audio Service for handling audio recording, playback, and processing
class AudioService {
    constructor() {
        this.mediaRecorder = null;
        this.audioStream = null;
        this.recordingStartTime = null;
        this.recordingTimerInterval = null;
        this.currentAudio = null;
        this.currentlyPlayingSegment = null;

        this.config = AppConfig.AUDIO;
    }

    // File validation
    isValidAudioFile(file) {
        if (!file) return false;

        const fileName = file.name.toLowerCase();
        const isValidExtension = this.config.SUPPORTED_FORMATS.some(format =>
            fileName.endsWith(format)
        );

        const isValidMimeType = this.config.MIME_TYPES.some(mimeType =>
            file.type === mimeType
        );

        return isValidExtension || isValidMimeType;
    }

    getFileValidationError(file) {
        if (!file) return 'No file selected';

        if (!this.isValidAudioFile(file)) {
            return `Please select a valid audio file (${this.config.SUPPORTED_FORMATS.join(', ')})`;
        }

        if (file.size > AppConfig.UI.MAX_FILE_SIZE) {
            return `File too large. Maximum size is ${this.formatFileSize(AppConfig.UI.MAX_FILE_SIZE)}`;
        }

        return null;
    }

    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // Recording functionality
    async startRecording() {
        try {
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: this.config.RECORDING
            });

            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            const audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const wavBlob = await this.convertToWav(audioBlob);
                this.emitRecordingComplete(wavBlob);
            };

            this.mediaRecorder.start();
            this.recordingStartTime = Date.now();
            this.startRecordingTimer();

            this.emitRecordingStarted();

        } catch (error) {
            console.error('Error accessing microphone:', error);
            throw new Error('Could not access microphone. Please check permissions.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.audioStream.getTracks().forEach(track => track.stop());
            this.stopRecordingTimer();
            this.emitRecordingStopped();
        }
    }

    isRecording() {
        return this.mediaRecorder && this.mediaRecorder.state === 'recording';
    }

    startRecordingTimer() {
        this.recordingTimerInterval = setInterval(() => {
            const elapsed = Date.now() - this.recordingStartTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            const timeString = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

            this.emitTimerUpdate(timeString, elapsed);
        }, 1000);
    }

    stopRecordingTimer() {
        if (this.recordingTimerInterval) {
            clearInterval(this.recordingTimerInterval);
            this.recordingTimerInterval = null;
        }
    }

    // Audio conversion
    async convertToWav(audioBlob) {
        return new Promise((resolve) => {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const fileReader = new FileReader();

            fileReader.onload = async function(e) {
                try {
                    const arrayBuffer = e.target.result;
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    const wavBlob = this.audioBufferToWav(audioBuffer);
                    resolve(wavBlob);
                } catch (error) {
                    console.error('Error converting audio:', error);
                    resolve(audioBlob); // Fallback to original
                }
            }.bind(this);

            fileReader.readAsArrayBuffer(audioBlob);
        });
    }

    audioBufferToWav(buffer) {
        const length = buffer.length;
        const sampleRate = buffer.sampleRate;
        const numberOfChannels = buffer.numberOfChannels;
        const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2);
        const view = new DataView(arrayBuffer);

        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length * numberOfChannels * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numberOfChannels * 2, true);
        view.setUint16(32, numberOfChannels * 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, length * numberOfChannels * 2, true);

        let offset = 44;
        for (let i = 0; i < length; i++) {
            for (let channel = 0; channel < numberOfChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }

    // Audio playback
    async playAudioFile(audioInput) {
        if (this.currentAudio) {
            this.stopCurrentAudio();
        }

        try {
            let url;
            if (audioInput instanceof Blob) {
                url = URL.createObjectURL(audioInput);
            } else if (audioInput instanceof File) {
                url = URL.createObjectURL(audioInput);
            } else {
                throw new Error('Invalid audio input for playback');
            }

            this.currentAudio = new Audio(url);
            await this.currentAudio.play();

            this.currentAudio.addEventListener('ended', () => {
                this.stopCurrentAudio();
            });

        } catch (error) {
            console.error('Error playing audio:', error);
            throw new Error('Could not play audio file');
        }
    }

    async playAudioSegment(audioInput, startTime, endTime, segmentElement = null) {
        if (this.currentAudio) {
            this.stopCurrentAudio();
        }

        try {
            let url;
            if (audioInput instanceof Blob) {
                url = URL.createObjectURL(audioInput);
            } else if (audioInput instanceof File) {
                url = URL.createObjectURL(audioInput);
            } else {
                throw new Error('Invalid audio input for segment playback');
            }

            this.currentAudio = new Audio(url);
            this.currentlyPlayingSegment = segmentElement;

            // Set up event listeners
            const handleTimeUpdate = () => {
                if (this.currentAudio.currentTime >= endTime) {
                    this.stopCurrentAudio();
                }
            };

            const handleEnded = () => {
                this.stopCurrentAudio();
            };

            this.currentAudio.addEventListener('timeupdate', handleTimeUpdate);
            this.currentAudio.addEventListener('ended', handleEnded);

            // Store cleanup function
            this.currentAudio._cleanup = () => {
                this.currentAudio.removeEventListener('timeupdate', handleTimeUpdate);
                this.currentAudio.removeEventListener('ended', handleEnded);
            };

            // Set start time and play
            this.currentAudio.currentTime = startTime;
            await this.currentAudio.play();

            this.emitSegmentPlaybackStarted(startTime, endTime, segmentElement);

        } catch (error) {
            console.error('Error playing audio segment:', error);
            this.stopCurrentAudio();
            throw new Error('Could not play audio segment');
        }
    }

    stopCurrentAudio() {
        if (this.currentAudio) {
            this.currentAudio.pause();

            if (this.currentAudio._cleanup) {
                this.currentAudio._cleanup();
                delete this.currentAudio._cleanup;
            }

            this.currentAudio = null;
        }

        if (this.currentlyPlayingSegment) {
            this.emitSegmentPlaybackStopped(this.currentlyPlayingSegment);
            this.currentlyPlayingSegment = null;
        }
    }

    isPlaying() {
        return this.currentAudio && !this.currentAudio.paused;
    }

    // Event emitters
    emitRecordingStarted() {
        const event = new CustomEvent('recordingStarted');
        document.dispatchEvent(event);
    }

    emitRecordingStopped() {
        const event = new CustomEvent('recordingStopped');
        document.dispatchEvent(event);
    }

    emitRecordingComplete(audioBlob) {
        const event = new CustomEvent('recordingComplete', {
            detail: { audioBlob }
        });
        document.dispatchEvent(event);
    }

    emitTimerUpdate(timeString, elapsed) {
        const event = new CustomEvent('recordingTimerUpdate', {
            detail: { timeString, elapsed }
        });
        document.dispatchEvent(event);
    }

    emitSegmentPlaybackStarted(startTime, endTime, element) {
        const event = new CustomEvent('segmentPlaybackStarted', {
            detail: { startTime, endTime, element }
        });
        document.dispatchEvent(event);
    }

    emitSegmentPlaybackStopped(element) {
        const event = new CustomEvent('segmentPlaybackStopped', {
            detail: { element }
        });
        document.dispatchEvent(event);
    }

    // Cleanup
    cleanup() {
        this.stopRecording();
        this.stopCurrentAudio();
        this.stopRecordingTimer();
    }
}
