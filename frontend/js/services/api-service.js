// API Service for handling all backend communication
class ApiService {
    constructor() {
        this.baseURL = AppConfig.API.BASE_URL;
        this.endpoints = AppConfig.API.ENDPOINTS;
        this.timeout = AppConfig.API.TIMEOUT;
        this.healthCheckInterval = null;
    }

    // Health monitoring
    async checkHealth() {
        try {
            const response = await axios.get(`${this.baseURL}${this.endpoints.HEALTH}`);

            if (response.data.status === 'healthy') {
                const statusInfo = this.formatHealthStatus(response.data);
                this.emitHealthStatus('healthy', statusInfo);
                return true;
            } else {
                throw new Error('API not healthy');
            }
        } catch (error) {
            console.error('API health check failed:', error);
            const errorMessage = this.formatHealthError(error);
            this.emitHealthStatus('unhealthy', errorMessage);
            return false;
        }
    }

    formatHealthStatus(data) {
        const device = data.device || 'Unknown';
        const gpuAvailable = data.gpu_available || false;
        const gpuName = data.gpu_name || 'N/A';

        if (gpuAvailable && gpuName !== 'N/A') {
            return `API Online (GPU: ${gpuName})`;
        } else {
            return `API Online (${device.toUpperCase()})`;
        }
    }

    formatHealthError(error) {
        if (error.response && error.response.status === 503) {
            return 'API Starting Up...';
        }
        return 'API Offline';
    }

    emitHealthStatus(status, message) {
        const event = new CustomEvent('apiHealthChanged', {
            detail: { status, message }
        });
        document.dispatchEvent(event);
    }

    startHealthMonitoring() {
        // Initial check
        this.checkHealth();

        // Set up interval
        this.healthCheckInterval = setInterval(() => {
            this.checkHealth();
        }, AppConfig.UI.HEALTH_CHECK_INTERVAL);
    }

    stopHealthMonitoring() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }
    }

    // GPU information
    async getGpuInfo() {
        try {
            const response = await axios.get(`${this.baseURL}${this.endpoints.GPU_INFO}`);
            return response.data;
        } catch (error) {
            console.error('Failed to get GPU info:', error);
            throw error;
        }
    }

    // Transcription methods
    async transcribe(audioInput, settings = {}) {
        const formData = this.createFormData(audioInput, settings);

        try {
            const response = await axios.post(
                `${this.baseURL}${this.endpoints.TRANSCRIBE}`,
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    timeout: this.timeout,
                    onUploadProgress: (progressEvent) => {
                        const progress = (progressEvent.loaded / progressEvent.total) * 100;
                        this.emitUploadProgress(progress);
                    }
                }
            );

            return response.data;
        } catch (error) {
            console.error('Transcription failed:', error);
            throw error;
        }
    }

    async diarize(audioInput, settings = {}) {
        const formData = this.createFormData(audioInput, settings);

        try {
            const response = await axios.post(
                `${this.baseURL}${this.endpoints.DIARIZE}`,
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    timeout: this.timeout,
                    onUploadProgress: (progressEvent) => {
                        const progress = (progressEvent.loaded / progressEvent.total) * 100;
                        this.emitUploadProgress(progress);
                    }
                }
            );

            return response.data;
        } catch (error) {
            console.error('Diarization failed:', error);
            throw error;
        }
    }

    async transcribeAndDiarize(audioInput, settings = {}) {
        const formData = this.createFormData(audioInput, settings);

        try {
            const response = await axios.post(
                `${this.baseURL}${this.endpoints.TRANSCRIBE_AND_DIARIZE}`,
                formData,
                {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    timeout: this.timeout,
                    onUploadProgress: (progressEvent) => {
                        const progress = (progressEvent.loaded / progressEvent.total) * 100;
                        this.emitUploadProgress(progress);
                    }
                }
            );

            return response.data;
        } catch (error) {
            console.error('Combined transcription and diarization failed:', error);
            throw error;
        }
    }

    // Helper methods
    createFormData(audioInput, settings) {
        const formData = new FormData();

        // Add audio file
        if (audioInput instanceof Blob) {
            formData.append('file', audioInput, 'recording.wav');
        } else if (audioInput instanceof File) {
            formData.append('file', audioInput);
        } else {
            throw new Error('Invalid audio input type');
        }

        // Add settings
        if (settings.returnTimestamps !== undefined) {
            formData.append('return_timestamps', settings.returnTimestamps);
        }

        if (settings.numSpeakers !== undefined && settings.numSpeakers > 0) {
            formData.append('num_speakers', settings.numSpeakers);
        }

        if (settings.chunkDurationSec !== undefined) {
            formData.append('chunk_duration_sec', settings.chunkDurationSec);
        }

        if (settings.overlapDurationSec !== undefined) {
            formData.append('overlap_duration_sec', settings.overlapDurationSec);
        }

        return formData;
    }

    emitUploadProgress(progress) {
        const event = new CustomEvent('uploadProgress', {
            detail: { progress }
        });
        document.dispatchEvent(event);
    }

    // Batch processing (future feature)
    async processBatch(audioFiles, settings = {}) {
        const results = [];

        for (let i = 0; i < audioFiles.length; i++) {
            const file = audioFiles[i];

            try {
                this.emitBatchProgress(i + 1, audioFiles.length, file.name);

                const result = await this.transcribeAndDiarize(file, settings);
                results.push({ file: file.name, success: true, data: result });

            } catch (error) {
                console.error(`Batch processing failed for ${file.name}:`, error);
                results.push({ file: file.name, success: false, error: error.message });
            }
        }

        return results;
    }

    emitBatchProgress(current, total, filename) {
        const event = new CustomEvent('batchProgress', {
            detail: { current, total, filename }
        });
        document.dispatchEvent(event);
    }
}
