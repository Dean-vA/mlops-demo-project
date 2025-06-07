// Audio Utility Functions
class AudioUtils {
    // Audio format detection and validation
    static isValidAudioFormat(file) {
        const validFormats = AppConfig.AUDIO.SUPPORTED_FORMATS;
        const fileName = file.name.toLowerCase();
        return validFormats.some(format => fileName.endsWith(format));
    }

    static getAudioFormat(file) {
        const extension = file.name.split('.').pop().toLowerCase();
        const formatMap = {
            'wav': 'audio/wav',
            'flac': 'audio/flac',
            'mp3': 'audio/mpeg',
            'ogg': 'audio/ogg',
            'webm': 'audio/webm',
            'm4a': 'audio/mp4'
        };
        return formatMap[extension] || 'audio/wav';
    }

    // Audio duration calculation
    static async getAudioDuration(file) {
        return new Promise((resolve, reject) => {
            const audio = new Audio();
            const url = URL.createObjectURL(file);

            const cleanup = () => {
                URL.revokeObjectURL(url);
                audio.removeEventListener('loadedmetadata', onLoad);
                audio.removeEventListener('error', onError);
            };

            const onLoad = () => {
                cleanup();
                resolve(audio.duration);
            };

            const onError = () => {
                cleanup();
                reject(new Error('Could not load audio file'));
            };

            audio.addEventListener('loadedmetadata', onLoad);
            audio.addEventListener('error', onError);
            audio.src = url;
        });
    }

    // Audio analysis utilities
    static async analyzeAudioFile(file) {
        try {
            const duration = await this.getAudioDuration(file);
            const sampleRate = await this.getAudioSampleRate(file);
            const channels = await this.getAudioChannels(file);

            return {
                duration,
                sampleRate,
                channels,
                bitrate: this.estimateBitrate(file.size, duration),
                size: file.size,
                format: this.getAudioFormat(file)
            };
        } catch (error) {
            console.error('Error analyzing audio file:', error);
            return null;
        }
    }

    static async getAudioSampleRate(file) {
        // This is a simplified estimation - for accurate sample rate,
        // you'd need to parse the audio file headers
        const format = this.getAudioFormat(file);

        // Common sample rates by format
        const defaultSampleRates = {
            'audio/wav': 44100,
            'audio/flac': 44100,
            'audio/mp3': 44100,
            'audio/ogg': 44100,
            'audio/webm': 48000
        };

        return defaultSampleRates[format] || 44100;
    }

    static async getAudioChannels(file) {
        // This is a simplified estimation
        // For accurate channel count, you'd need to parse the audio file
        return 1; // Assume mono for STT
    }

    static estimateBitrate(fileSize, duration) {
        if (!duration || duration === 0) return 0;
        // Rough bitrate estimation: (file size in bits) / duration
        return Math.round((fileSize * 8) / duration);
    }

    // Audio conversion utilities
    static async convertToWav(audioBlob, sampleRate = 44100) {
        return new Promise((resolve, reject) => {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: sampleRate
            });

            const fileReader = new FileReader();

            fileReader.onload = async function(e) {
                try {
                    const arrayBuffer = e.target.result;
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    const wavBlob = AudioUtils.audioBufferToWav(audioBuffer);
                    resolve(wavBlob);
                } catch (error) {
                    console.error('Error converting audio:', error);
                    resolve(audioBlob); // Fallback to original
                }
            };

            fileReader.onerror = () => {
                reject(new Error('Failed to read audio file'));
            };

            fileReader.readAsArrayBuffer(audioBlob);
        });
    }

    static audioBufferToWav(buffer) {
        const length = buffer.length;
        const sampleRate = buffer.sampleRate;
        const numberOfChannels = buffer.numberOfChannels;

        // Create WAV file structure
        const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2);
        const view = new DataView(arrayBuffer);

        // WAV header
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

        // Convert float32 audio data to 16-bit PCM
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

    // Audio playback utilities
    static createAudioPlayer(source) {
        const audio = new Audio();

        if (source instanceof Blob || source instanceof File) {
            audio.src = URL.createObjectURL(source);
        } else if (typeof source === 'string') {
            audio.src = source;
        }

        return audio;
    }

    static async playAudioSegment(audioSource, startTime, endTime) {
        return new Promise((resolve, reject) => {
            const audio = this.createAudioPlayer(audioSource);

            const cleanup = () => {
                audio.removeEventListener('timeupdate', onTimeUpdate);
                audio.removeEventListener('ended', onEnded);
                audio.removeEventListener('error', onError);
                if (audio.src.startsWith('blob:')) {
                    URL.revokeObjectURL(audio.src);
                }
            };

            const onTimeUpdate = () => {
                if (audio.currentTime >= endTime) {
                    audio.pause();
                    cleanup();
                    resolve();
                }
            };

            const onEnded = () => {
                cleanup();
                resolve();
            };

            const onError = () => {
                cleanup();
                reject(new Error('Audio playback error'));
            };

            audio.addEventListener('timeupdate', onTimeUpdate);
            audio.addEventListener('ended', onEnded);
            audio.addEventListener('error', onError);

            audio.currentTime = startTime;
            audio.play().catch(reject);
        });
    }

    // Audio visualization utilities
    static async createWaveform(audioFile, width = 800, height = 100) {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const arrayBuffer = await FileUtils.readFileAsArrayBuffer(audioFile);
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        const data = audioBuffer.getChannelData(0);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;

        ctx.fillStyle = '#667eea';
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < width; i++) {
            let min = 1.0;
            let max = -1.0;

            for (let j = 0; j < step; j++) {
                const datum = data[(i * step) + j];
                if (datum < min) min = datum;
                if (datum > max) max = datum;
            }

            ctx.fillRect(i, (1 + min) * amp, 1, Math.max(1, (max - min) * amp));
        }

        return canvas;
    }

    // Audio quality assessment
    static analyzeAudioQuality(file, duration) {
        const bitrate = this.estimateBitrate(file.size, duration);
        const sizePerSecond = file.size / duration;

        let quality = 'Unknown';

        if (bitrate >= 320000) {
            quality = 'High';
        } else if (bitrate >= 192000) {
            quality = 'Good';
        } else if (bitrate >= 128000) {
            quality = 'Medium';
        } else if (bitrate >= 64000) {
            quality = 'Low';
        } else {
            quality = 'Very Low';
        }

        return {
            quality,
            bitrate,
            sizePerSecond,
            estimatedSampleRate: this.estimateSampleRate(bitrate),
            suitable: bitrate >= 64000 // Minimum for decent STT
        };
    }

    static estimateSampleRate(bitrate) {
        // Rough estimation based on common bitrate/sample rate combinations
        if (bitrate >= 256000) return 48000;
        if (bitrate >= 128000) return 44100;
        if (bitrate >= 64000) return 22050;
        return 16000;
    }

    // Audio processing utilities
    static normalizeAudio(audioBuffer) {
        const data = audioBuffer.getChannelData(0);
        let max = 0;

        // Find the maximum amplitude
        for (let i = 0; i < data.length; i++) {
            max = Math.max(max, Math.abs(data[i]));
        }

        // Normalize if needed
        if (max > 0 && max < 1) {
            const scale = 0.95 / max; // Leave some headroom
            for (let i = 0; i < data.length; i++) {
                data[i] *= scale;
            }
        }

        return audioBuffer;
    }

    static applyNoiseGate(audioBuffer, threshold = 0.01) {
        const data = audioBuffer.getChannelData(0);

        for (let i = 0; i < data.length; i++) {
            if (Math.abs(data[i]) < threshold) {
                data[i] = 0;
            }
        }

        return audioBuffer;
    }

    // Recording utilities
    static async getRecordingDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'audioinput');
        } catch (error) {
            console.error('Error getting recording devices:', error);
            return [];
        }
    }

    static async requestMicrophonePermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch (error) {
            console.error('Microphone permission denied:', error);
            return false;
        }
    }

    static getOptimalRecordingSettings() {
        return {
            audio: {
                sampleRate: AppConfig.AUDIO.RECORDING.SAMPLE_RATE,
                channelCount: AppConfig.AUDIO.RECORDING.CHANNEL_COUNT,
                echoCancellation: AppConfig.AUDIO.RECORDING.ECHO_CANCELLATION,
                noiseSuppression: AppConfig.AUDIO.RECORDING.NOISE_SUPPRESSION,
                autoGainControl: true,
                latency: 0.1 // 100ms latency
            }
        };
    }

    // Utility functions
    static formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
    }

    static formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(1);
        return `${minutes}:${secs.padStart(4, '0')}`;
    }

    static parseTimeString(timeString) {
        const parts = timeString.split(':');
        if (parts.length === 2) {
            return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
        } else if (parts.length === 3) {
            return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
        }
        return 0;
    }

    // Browser compatibility checks
    static checkAudioSupport() {
        const support = {
            webAudio: !!(window.AudioContext || window.webkitAudioContext),
            mediaRecorder: !!window.MediaRecorder,
            getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
            audioElement: !!document.createElement('audio').canPlayType
        };

        support.fullSupport = Object.values(support).every(Boolean);

        return support;
    }

    static getSupportedMimeTypes() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4',
            'audio/wav'
        ];

        return types.filter(type => MediaRecorder.isTypeSupported(type));
    }
}
