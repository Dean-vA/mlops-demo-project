// Download Component - Handles transcript downloads in multiple formats
class DownloadComponent {
    constructor(state) {
        this.state = state;
        this.container = document.getElementById('download-component');
    }

    async init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="download-section" id="download-section" style="display: none;">
                <h3>📥 Download Transcript</h3>
                <p>Download your transcript in different formats</p>
                <div class="download-buttons">
                    <button class="download-btn" id="download-txt-btn">
                        📄 Plain Text (.txt)
                    </button>
                    <button class="download-btn" id="download-csv-btn">
                        📊 CSV with Speakers (.csv)
                    </button>
                    <button class="download-btn" id="download-json-btn">
                        🔧 JSON Data (.json)
                    </button>
                    <button class="download-btn" id="download-srt-btn">
                        🎬 Subtitle File (.srt)
                    </button>
                </div>

                <div class="download-options" style="margin-top: 20px;">
                    <details>
                        <summary style="cursor: pointer; font-weight: 600;">📋 Advanced Options</summary>
                        <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin-top: 10px;">
                            <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                                <input type="checkbox" id="include-timestamps" checked>
                                Include timestamps in text format
                            </label>
                            <label style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                                <input type="checkbox" id="include-confidence" checked>
                                Include confidence scores (where available)
                            </label>
                            <label style="display: flex; align-items: center; gap: 8px;">
                                <input type="checkbox" id="use-edited-names" checked>
                                Use edited speaker names
                            </label>
                        </div>
                    </details>
                </div>
            </div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            downloadSection: this.container.querySelector('#download-section'),
            txtBtn: this.container.querySelector('#download-txt-btn'),
            csvBtn: this.container.querySelector('#download-csv-btn'),
            jsonBtn: this.container.querySelector('#download-json-btn'),
            srtBtn: this.container.querySelector('#download-srt-btn'),

            // Options
            includeTimestamps: this.container.querySelector('#include-timestamps'),
            includeConfidence: this.container.querySelector('#include-confidence'),
            useEditedNames: this.container.querySelector('#use-edited-names')
        };
    }

    setupEventListeners() {
        this.elements.txtBtn.addEventListener('click', () => this.downloadTranscript('txt'));
        this.elements.csvBtn.addEventListener('click', () => this.downloadTranscript('csv'));
        this.elements.jsonBtn.addEventListener('click', () => this.downloadTranscript('json'));
        this.elements.srtBtn.addEventListener('click', () => this.downloadTranscript('srt'));
    }

    downloadTranscript(format) {
        if (!this.state.currentTranscriptionData) {
            UIUtils.showToast('No transcription data available for download', 'error');
            return;
        }

        try {
            const options = this.getDownloadOptions();
            let content = '';
            let filename = '';
            let mimeType = '';

            const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');

            switch (format) {
                case 'txt':
                    content = this.generatePlainText(options);
                    filename = `transcript_${timestamp}.txt`;
                    mimeType = 'text/plain';
                    break;
                case 'csv':
                    content = this.generateCSV(options);
                    filename = `transcript_${timestamp}.csv`;
                    mimeType = 'text/csv';
                    break;
                case 'json':
                    content = this.generateJSON(options);
                    filename = `transcript_${timestamp}.json`;
                    mimeType = 'application/json';
                    break;
                case 'srt':
                    content = this.generateSRT(options);
                    filename = `transcript_${timestamp}.srt`;
                    mimeType = 'text/plain';
                    break;
                default:
                    throw new Error('Unsupported format');
            }

            this.triggerDownload(content, filename, mimeType);
            UIUtils.showToast(`Downloaded ${filename}`, 'success');

        } catch (error) {
            console.error('Download error:', error);
            UIUtils.showToast('Failed to generate download: ' + error.message, 'error');
        }
    }

    getDownloadOptions() {
        return {
            includeTimestamps: this.elements.includeTimestamps.checked,
            includeConfidence: this.elements.includeConfidence.checked,
            useEditedNames: this.elements.useEditedNames.checked
        };
    }

    generatePlainText(options) {
        const data = this.state.currentTranscriptionData;
        const transcriptionData = this.getTranscriptionData(data);
        const hasSpeakers = this.state.currentMode === 'diarize' && data.diarization;

        let content = '';

        // Header
        content += `TRANSCRIPT\n`;
        content += `Generated: ${new Date().toLocaleString()}\n`;
        content += `Processing Time: ${this.getProcessingTime(transcriptionData, data)}s\n`;
        content += `Word Count: ${this.getWordCount(transcriptionData)}\n`;

        if (hasSpeakers) {
            content += `Speakers Detected: ${data.diarization.num_speakers_detected || 0}\n`;
        }

        content += `\n${'='.repeat(50)}\n\n`;

        // Content
        if (transcriptionData.segments && transcriptionData.segments.length > 0) {
            // Segment-by-segment format
            transcriptionData.segments.forEach((segment, index) => {
                let line = '';

                if (options.includeTimestamps) {
                    const startTime = this.formatTime(segment.start);
                    const endTime = this.formatTime(segment.end);
                    line += `[${startTime} - ${endTime}] `;
                }

                if (hasSpeakers && segment.speaker) {
                    const speakerName = options.useEditedNames
                        ? this.state.getDisplayName(segment.speaker)
                        : segment.speaker;
                    line += `${speakerName.toUpperCase()}: `;
                }

                line += segment.text;

                if (options.includeConfidence && segment.speaker_confidence !== undefined) {
                    line += ` (confidence: ${(segment.speaker_confidence * 100).toFixed(1)}%)`;
                }

                content += line + '\n\n';
            });
        } else {
            // Plain text format
            const text = this.getTranscriptionText(transcriptionData);
            content += text;
        }

        return content;
    }

    generateCSV(options) {
        const data = this.state.currentTranscriptionData;
        const transcriptionData = this.getTranscriptionData(data);
        const hasSpeakers = this.state.currentMode === 'diarize' && data.diarization;

        let content = '';

        // CSV Header
        const headers = ['Start Time', 'End Time', 'Duration'];
        if (hasSpeakers) headers.push('Speaker');
        if (options.includeConfidence) headers.push('Confidence');
        headers.push('Text');

        content += headers.join(',') + '\n';

        // CSV Content
        if (transcriptionData.segments && transcriptionData.segments.length > 0) {
            transcriptionData.segments.forEach(segment => {
                const row = [];

                row.push(segment.start.toFixed(2));
                row.push(segment.end.toFixed(2));
                row.push((segment.end - segment.start).toFixed(2));

                if (hasSpeakers) {
                    const speakerName = options.useEditedNames && segment.speaker
                        ? this.state.getDisplayName(segment.speaker)
                        : (segment.speaker || 'Unknown');
                    row.push(`"${speakerName}"`);
                }

                if (options.includeConfidence) {
                    const confidence = segment.speaker_confidence !== undefined
                        ? (segment.speaker_confidence * 100).toFixed(1)
                        : 'N/A';
                    row.push(confidence);
                }

                const text = `"${segment.text.replace(/"/g, '""')}"`;
                row.push(text);

                content += row.join(',') + '\n';
            });
        } else {
            // Fallback for plain text
            const text = this.getTranscriptionText(transcriptionData);
            const row = ['0.00', '0.00', '0.00'];
            if (hasSpeakers) row.push('"Unknown"');
            if (options.includeConfidence) row.push('N/A');
            row.push(`"${text.replace(/"/g, '""')}"`);
            content += row.join(',') + '\n';
        }

        return content;
    }

    generateJSON(options) {
        const data = this.state.currentTranscriptionData;

        // Create a clean copy of the data
        const exportData = {
            metadata: {
                generated: new Date().toISOString(),
                format_version: '1.0',
                processing_time_sec: this.getProcessingTime(this.getTranscriptionData(data), data),
                word_count: this.getWordCount(this.getTranscriptionData(data)),
                options: options
            },
            transcription: {},
            diarization: null
        };

        // Add transcription data
        const transcriptionData = this.getTranscriptionData(data);
        exportData.transcription = {
            text: this.getTranscriptionText(transcriptionData),
            segments: transcriptionData.segments || [],
            timestamps: transcriptionData.timestamps || { word: [], segment: [] }
        };

        // Add diarization data if available
        if (this.state.currentMode === 'diarize' && data.diarization) {
            exportData.diarization = data.diarization;

            // Add edited speaker names if requested
            if (options.useEditedNames) {
                exportData.speaker_names = this.state.speakerNames;
            }
        }

        return JSON.stringify(exportData, null, 2);
    }

    generateSRT(options) {
        const data = this.state.currentTranscriptionData;
        const transcriptionData = this.getTranscriptionData(data);
        const hasSpeakers = this.state.currentMode === 'diarize' && data.diarization;

        if (!transcriptionData.segments || transcriptionData.segments.length === 0) {
            throw new Error('No segments available for SRT generation');
        }

        let content = '';

        transcriptionData.segments.forEach((segment, index) => {
            const startTime = this.formatSRTTime(segment.start);
            const endTime = this.formatSRTTime(segment.end);

            let text = segment.text;

            if (hasSpeakers && segment.speaker) {
                const speakerName = options.useEditedNames
                    ? this.state.getDisplayName(segment.speaker)
                    : segment.speaker;
                text = `${speakerName.toUpperCase()}: ${text}`;
            }

            content += `${index + 1}\n`;
            content += `${startTime} --> ${endTime}\n`;
            content += `${text}\n\n`;
        });

        return content;
    }

    // Helper methods
    getTranscriptionData(data) {
        return this.state.currentMode === 'diarize' && data.transcription
            ? data.transcription
            : data;
    }

    getTranscriptionText(transcriptionData) {
        if (typeof transcriptionData.text === 'string') {
            return transcriptionData.text;
        } else if (Array.isArray(transcriptionData.text) && transcriptionData.text.length > 0) {
            return transcriptionData.text[0].text || 'No text available';
        }
        return 'No text available';
    }

    getProcessingTime(transcriptionData, rawData) {
        return transcriptionData.processing_time_sec ||
               rawData.combined_processing_time || 0;
    }

    getWordCount(transcriptionData) {
        const text = this.getTranscriptionText(transcriptionData);
        return text.split(' ').filter(word => word.trim().length > 0).length;
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(1);
        return `${minutes}:${secs.padStart(4, '0')}`;
    }

    formatSRTTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        const milliseconds = Math.floor((secs % 1) * 1000);

        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${Math.floor(secs).toString().padStart(2, '0')},${milliseconds.toString().padStart(3, '0')}`;
    }

    triggerDownload(content, filename, mimeType) {
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

    show() {
        this.elements.downloadSection.style.display = 'block';
        UIUtils.slideDown(this.elements.downloadSection);
    }

    hide() {
        this.elements.downloadSection.style.display = 'none';
    }
}
