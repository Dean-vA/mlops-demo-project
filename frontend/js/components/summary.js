// Summary Component - Handles D&D session summary generation and display
class SummaryComponent {
    constructor(state, services) {
        this.state = state;
        this.services = services;
        this.container = document.getElementById('summary-component');
        this.currentSummary = null;
        this.isGenerating = false;
        this.isReady = false; // Track if component is ready for summary generation
    }

    async init() {
        this.render();
        this.setupEventListeners();
    }

    render() {
        this.container.innerHTML = `
            <div class="summary-tab-content" id="summary-tab-content" style="display: none;">
                <div class="summary-header">
                    <h3>🐉 D&D Session Summary</h3>
                    <p class="summary-description">Generate an AI-powered narrative summary of your D&D session, including story progression, character moments, and epic encounters.</p>
                </div>

                <div class="summary-controls">
                    <button class="btn btn-primary" id="generate-summary-btn" disabled>
                        ✨ Generate Session Summary
                    </button>

                    <div class="summary-info" id="summary-info" style="display: none;">
                        <div class="summary-stats">
                            <span id="chunk-count">-</span> chunks to process
                            <span id="word-count-estimate">~0</span> words
                        </div>
                    </div>
                </div>

                <div class="summary-progress" id="summary-progress" style="display: none;">
                    <div class="progress-header">
                        <span id="progress-text">Initializing...</span>
                        <span id="progress-percentage">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <div class="progress-details" id="progress-details"></div>
                </div>

                <div class="summary-result" id="summary-result" style="display: none;">
                    <div class="summary-sections" id="summary-sections"></div>

                    <div class="summary-metadata">
                        <div class="summary-stats-grid">
                            <div class="stat-item">
                                <span class="stat-value" id="final-word-count">-</span>
                                <span class="stat-label">Summary Words</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value" id="chunks-processed">-</span>
                                <span class="stat-label">Chunks Processed</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value" id="generation-time">-</span>
                                <span class="stat-label">Generation Time</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value" id="original-words">-</span>
                                <span class="stat-label">Original Words</span>
                            </div>
                        </div>

                        <!-- Token Usage Stats -->
                        <div class="token-usage-section">
                            <h4 class="token-usage-title">🪙 Token Usage & Cost Analysis</h4>
                            <div class="token-stats-grid">
                                <div class="token-stat-item">
                                    <span class="token-stat-value" id="total-tokens">-</span>
                                    <span class="token-stat-label">Total Tokens</span>
                                </div>
                                <div class="token-stat-item">
                                    <span class="token-stat-value" id="prompt-tokens">-</span>
                                    <span class="token-stat-label">Input Tokens</span>
                                </div>
                                <div class="token-stat-item">
                                    <span class="token-stat-value" id="completion-tokens">-</span>
                                    <span class="token-stat-label">Output Tokens</span>
                                </div>
                                <div class="token-stat-item cost-highlight">
                                    <span class="token-stat-value" id="estimated-cost">$-</span>
                                    <span class="token-stat-label">Estimated Cost</span>
                                </div>
                            </div>

                            <div class="efficiency-metrics">
                                <div class="efficiency-item">
                                    <span class="efficiency-label">Compression Ratio:</span>
                                    <span class="efficiency-value" id="compression-ratio">-:1</span>
                                </div>
                                <div class="efficiency-item">
                                    <span class="efficiency-label">Processing Speed:</span>
                                    <span class="efficiency-value" id="processing-speed">- words/sec</span>
                                </div>
                                <div class="efficiency-item">
                                    <span class="efficiency-label">Tokens per Word:</span>
                                    <span class="efficiency-value" id="tokens-per-word">- tokens</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="summary-actions">
                        <button class="btn btn-secondary" id="copy-summary-btn">
                            📋 Copy Summary
                        </button>
                        <button class="btn btn-secondary" id="export-summary-btn">
                            📄 Export as Markdown
                        </button>
                        <button class="btn btn-primary" id="regenerate-summary-btn">
                            🔄 Regenerate Summary
                        </button>
                    </div>
                </div>

                <div class="summary-error" id="summary-error" style="display: none;">
                    <div class="error-content">
                        <span class="error-icon">⚠️</span>
                        <span class="error-text" id="summary-error-text"></span>
                    </div>
                    <button class="btn btn-primary" id="retry-summary-btn">
                        🔄 Try Again
                    </button>
                </div>
            </div>
        `;

        this.cacheElements();
    }

    cacheElements() {
        this.elements = {
            tabContent: this.container.querySelector('#summary-tab-content'),
            generateBtn: this.container.querySelector('#generate-summary-btn'),
            summaryInfo: this.container.querySelector('#summary-info'),
            chunkCount: this.container.querySelector('#chunk-count'),
            wordCountEstimate: this.container.querySelector('#word-count-estimate'),

            // Progress elements
            progressContainer: this.container.querySelector('#summary-progress'),
            progressText: this.container.querySelector('#progress-text'),
            progressPercentage: this.container.querySelector('#progress-percentage'),
            progressFill: this.container.querySelector('#progress-fill'),
            progressDetails: this.container.querySelector('#progress-details'),

            // Result elements
            resultContainer: this.container.querySelector('#summary-result'),
            summarySections: this.container.querySelector('#summary-sections'),
            finalWordCount: this.container.querySelector('#final-word-count'),
            chunksProcessed: this.container.querySelector('#chunks-processed'),
            generationTime: this.container.querySelector('#generation-time'),
            originalWords: this.container.querySelector('#original-words'),

            // Token usage elements
            totalTokens: this.container.querySelector('#total-tokens'),
            promptTokens: this.container.querySelector('#prompt-tokens'),
            completionTokens: this.container.querySelector('#completion-tokens'),
            estimatedCost: this.container.querySelector('#estimated-cost'),
            compressionRatio: this.container.querySelector('#compression-ratio'),
            processingSpeed: this.container.querySelector('#processing-speed'),
            tokensPerWord: this.container.querySelector('#tokens-per-word'),

            // Action buttons
            copySummaryBtn: this.container.querySelector('#copy-summary-btn'),
            exportSummaryBtn: this.container.querySelector('#export-summary-btn'),
            regenerateBtn: this.container.querySelector('#regenerate-summary-btn'),

            // Error elements
            errorContainer: this.container.querySelector('#summary-error'),
            errorText: this.container.querySelector('#summary-error-text'),
            retryBtn: this.container.querySelector('#retry-summary-btn')
        };
    }

    setupEventListeners() {
        this.elements.generateBtn.addEventListener('click', () => {
            this.generateSummary();
        });

        this.elements.regenerateBtn.addEventListener('click', () => {
            this.generateSummary();
        });

        this.elements.retryBtn.addEventListener('click', () => {
            this.generateSummary();
        });

        this.elements.copySummaryBtn.addEventListener('click', () => {
            this.copySummary();
        });

        this.elements.exportSummaryBtn.addEventListener('click', () => {
            this.exportSummary();
        });

        // Listen for transcription completion
        this.state.on('transcriptionComplete', (data) => {
            this.onTranscriptionComplete(data);
        });
    }

    onTranscriptionComplete(data) {
        console.log('Summary component received transcription data:', data);

        // Check if this is diarization data with segments
        const isDiarization = this.state.currentMode === 'diarize' && data.diarization;

        if (isDiarization && data.transcription) {
            this.prepareSummaryGeneration(data);
            this.state.currentTranscriptionData = data; // Store for summary generation
            this.isReady = true;
        } else {
            this.isReady = false;
            // Only disable if not ready for summary
            this.elements.generateBtn.disabled = true;
        }
    }

    prepareSummaryGeneration(data) {
        console.log('Preparing summary generation...');

        // Estimate chunks and word count
        const segments = this.extractSegments(data.transcription);
        if (segments && segments.length > 0) {
            const wordCount = this.estimateWordCount(segments);
            const chunkCount = Math.max(1, Math.ceil(wordCount / 15000)); // 15k words per chunk

            this.elements.chunkCount.textContent = chunkCount;
            this.elements.wordCountEstimate.textContent = wordCount.toLocaleString();
            this.elements.summaryInfo.style.display = 'block';

            // Enable the generate button
            this.elements.generateBtn.disabled = false;
            this.isReady = true;

            console.log('Summary generation prepared - button enabled');
        } else {
            console.warn('No segments found for summary generation');
            this.elements.generateBtn.disabled = true;
            this.isReady = false;
        }
    }

    extractSegments(transcriptionData) {
        if (Array.isArray(transcriptionData.text) && transcriptionData.text.length > 0) {
            const firstItem = transcriptionData.text[0];
            if (firstItem.timestamp && firstItem.timestamp.segment) {
                return firstItem.timestamp.segment;
            }
        }

        // Also check segments directly
        if (transcriptionData.segments && transcriptionData.segments.length > 0) {
            return transcriptionData.segments;
        }

        return null;
    }

    estimateWordCount(segments) {
        let totalWords = 0;
        segments.forEach(segment => {
            const text = segment.segment || segment.text || '';
            totalWords += text.split(' ').length;
        });
        return totalWords;
    }

    async generateSummary() {
        if (this.isGenerating) return;

        try {
            this.isGenerating = true;
            this.showProgress();

            // Get current transcription data
            const transcriptionData = this.state.currentTranscriptionData;
            if (!transcriptionData || this.state.currentMode !== 'diarize') {
                throw new Error('No diarization data available for summarization');
            }

            // Extract segments
            const segments = this.extractSegments(transcriptionData.transcription);
            if (!segments || segments.length === 0) {
                throw new Error('No segments found in transcription data');
            }

            this.updateProgress('Preparing segments for summarization...', 0.05);

            // Prepare request data
            const requestData = {
                segments: segments,
                speaker_names: this.state.speakerNames
            };

            console.log('Sending summarization request:', requestData);

            // Make API request
            const response = await this.services.api.summarizeSession(requestData, (message, progress) => {
                this.updateProgress(message, progress);
            });

            console.log('Summary response:', response);

            // Display the summary
            this.displaySummary(response);

        } catch (error) {
            console.error('Summary generation error:', error);
            this.showError(this.formatError(error));
        } finally {
            this.isGenerating = false;
        }
    }

    showProgress() {
        this.elements.progressContainer.style.display = 'block';
        this.elements.resultContainer.style.display = 'none';
        this.elements.errorContainer.style.display = 'none';
        this.elements.generateBtn.disabled = true;
    }

    updateProgress(message, progress) {
        const percentage = Math.round(progress * 100);

        this.elements.progressText.textContent = message;
        this.elements.progressPercentage.textContent = `${percentage}%`;
        this.elements.progressFill.style.width = `${percentage}%`;

        // Add some detail messages
        if (progress < 0.2) {
            this.elements.progressDetails.textContent = 'Analyzing transcript structure...';
        } else if (progress < 0.8) {
            const currentChunk = Math.ceil((progress - 0.2) / 0.6 * parseInt(this.elements.chunkCount.textContent));
            const totalChunks = parseInt(this.elements.chunkCount.textContent);
            this.elements.progressDetails.textContent = `Processing chunk ${currentChunk} of ${totalChunks}...`;
        } else if (progress < 1.0) {
            this.elements.progressDetails.textContent = 'Creating final comprehensive summary...';
        } else {
            this.elements.progressDetails.textContent = 'Summary generation complete!';
        }
    }

    displaySummary(summaryData) {
        this.currentSummary = summaryData;

        // Hide progress, show results
        this.elements.progressContainer.style.display = 'none';
        this.elements.resultContainer.style.display = 'block';
        // Re-enable the generate button (for regeneration)
        this.elements.generateBtn.disabled = false;

        // Display sections
        this.displaySummarySections(summaryData.sections || {});

        // Display metadata
        this.elements.finalWordCount.textContent = (summaryData.word_count || 0).toLocaleString();
        this.elements.chunksProcessed.textContent = `${summaryData.chunks_successful || 0}/${summaryData.chunks_processed || 0}`;
        this.elements.generationTime.textContent = `${(summaryData.generation_time_sec || 0).toFixed(1)}s`;
        this.elements.originalWords.textContent = (summaryData.original_transcript_words || 0).toLocaleString();

        // Display token usage stats
        if (summaryData.token_usage) {
            const tokenUsage = summaryData.token_usage;

            this.elements.totalTokens.textContent = (tokenUsage.total_tokens || 0).toLocaleString();
            this.elements.promptTokens.textContent = (tokenUsage.prompt_tokens || 0).toLocaleString();
            this.elements.completionTokens.textContent = (tokenUsage.completion_tokens || 0).toLocaleString();
            this.elements.estimatedCost.textContent = `$${(tokenUsage.estimated_cost_usd?.total || 0).toFixed(4)}`;

            // Efficiency metrics
            if (tokenUsage.efficiency_metrics) {
                const efficiency = tokenUsage.efficiency_metrics;
                this.elements.compressionRatio.textContent = `${efficiency.compression_ratio || 0}:1`;
                this.elements.processingSpeed.textContent = `${(efficiency.processing_speed_words_per_sec || 0).toFixed(0)}`;
                this.elements.tokensPerWord.textContent = `${(efficiency.tokens_per_word_generated || 0).toFixed(1)}`;
            }
        }
    }

    displaySummarySections(sections) {
        const sectionOrder = [
            'Session Highlights',
            'Story Progression',
            'Combat & Challenges',
            'Character Moments',
            'World Building & NPCs',
            'Hooks & Next Steps'
        ];

        let sectionsHTML = '';

        sectionOrder.forEach(sectionName => {
            const content = sections[sectionName];
            if (content && content.trim()) {
                const emoji = this.getSectionEmoji(sectionName);
                sectionsHTML += `
                    <div class="summary-section-item">
                        <h4 class="summary-section-title">${emoji} ${sectionName}</h4>
                        <div class="summary-section-content">${content.replace(/\n/g, '<br>')}</div>
                    </div>
                `;
            }
        });

        if (!sectionsHTML) {
            sectionsHTML = `
                <div class="summary-section-item">
                    <h4 class="summary-section-title">📝 Complete Summary</h4>
                    <div class="summary-section-content">${(this.currentSummary?.summary || 'No summary content available').replace(/\n/g, '<br>')}</div>
                </div>
            `;
        }

        this.elements.summarySections.innerHTML = sectionsHTML;
    }

    getSectionEmoji(sectionName) {
        const emojiMap = {
            'Session Highlights': '🎭',
            'Story Progression': '📖',
            'Combat & Challenges': '⚔️',
            'Character Moments': '🎪',
            'World Building & NPCs': '🌍',
            'Hooks & Next Steps': '🎣'
        };
        return emojiMap[sectionName] || '📝';
    }

    showError(message) {
        this.elements.progressContainer.style.display = 'none';
        this.elements.resultContainer.style.display = 'none';
        this.elements.errorContainer.style.display = 'block';
        this.elements.errorText.textContent = message;
        // Re-enable button for retry
        this.elements.generateBtn.disabled = !this.isReady;
    }

    formatError(error) {
        if (error.response) {
            if (error.response.status === 503) {
                return 'Summarization not available. OpenAI API key not configured on the server.';
            } else {
                return `API Error: ${error.response.data.detail || error.response.statusText}`;
            }
        } else if (error.request) {
            return 'Network error: Could not connect to the API for summarization.';
        } else {
            return `Error: ${error.message}`;
        }
    }

    async copySummary() {
        if (!this.currentSummary) {
            UIUtils.showToast('No summary to copy', 'warning');
            return;
        }

        const summaryText = this.formatSummaryAsText(this.currentSummary);
        await UIUtils.copyToClipboard(summaryText);
    }

    exportSummary() {
        if (!this.currentSummary) {
            UIUtils.showToast('No summary to export', 'warning');
            return;
        }

        const markdown = this.formatSummaryAsMarkdown(this.currentSummary);
        const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
        const filename = `dnd-session-summary-${timestamp}.md`;

        FileUtils.downloadFile(markdown, filename, 'text/markdown');
        UIUtils.showToast('Summary exported as Markdown', 'success');
    }

    formatSummaryAsText(summaryData) {
        let text = 'D&D SESSION SUMMARY\n';
        text += '==================\n\n';

        const sections = summaryData.sections || {};
        const sectionOrder = ['Session Highlights', 'Story Progression', 'Combat & Challenges', 'Character Moments', 'World Building & NPCs', 'Hooks & Next Steps'];

        sectionOrder.forEach(sectionName => {
            const content = sections[sectionName];
            if (content && content.trim()) {
                text += `${sectionName.toUpperCase()}\n`;
                text += '-'.repeat(sectionName.length) + '\n';
                text += content + '\n\n';
            }
        });

        text += '\nSUMMARY STATISTICS\n';
        text += '==================\n';
        text += `Generated: ${new Date().toLocaleString()}\n`;
        text += `Original Transcript: ${(summaryData.original_transcript_words || 0).toLocaleString()} words\n`;
        text += `Summary Length: ${(summaryData.word_count || 0).toLocaleString()} words\n`;
        text += `Processing Time: ${(summaryData.generation_time_sec || 0).toFixed(1)} seconds\n`;
        text += `Chunks Processed: ${summaryData.chunks_successful || 0}/${summaryData.chunks_processed || 0}\n`;

        // Add token usage stats
        if (summaryData.token_usage) {
            const tokenUsage = summaryData.token_usage;
            text += '\nTOKEN USAGE & COSTS\n';
            text += '===================\n';
            text += `Total Tokens: ${(tokenUsage.total_tokens || 0).toLocaleString()}\n`;
            text += `Input Tokens: ${(tokenUsage.prompt_tokens || 0).toLocaleString()}\n`;
            text += `Output Tokens: ${(tokenUsage.completion_tokens || 0).toLocaleString()}\n`;
            text += `Estimated Cost: $${(tokenUsage.estimated_cost_usd?.total || 0).toFixed(4)}\n`;

            if (tokenUsage.efficiency_metrics) {
                const efficiency = tokenUsage.efficiency_metrics;
                text += `Compression Ratio: ${efficiency.compression_ratio || 0}:1\n`;
                text += `Processing Speed: ${(efficiency.processing_speed_words_per_sec || 0).toFixed(0)} words/sec\n`;
                text += `Tokens per Generated Word: ${(efficiency.tokens_per_word_generated || 0).toFixed(1)}\n`;
            }
        }

        return text;
    }

    formatSummaryAsMarkdown(summaryData) {
        let markdown = '# 🐉 D&D Session Summary\n\n';
        markdown += `*Generated on ${new Date().toLocaleString()}*\n\n`;

        const sections = summaryData.sections || {};
        const sectionOrder = ['Session Highlights', 'Story Progression', 'Combat & Challenges', 'Character Moments', 'World Building & NPCs', 'Hooks & Next Steps'];

        sectionOrder.forEach(sectionName => {
            const content = sections[sectionName];
            if (content && content.trim()) {
                const emoji = this.getSectionEmoji(sectionName);
                markdown += `## ${emoji} ${sectionName}\n\n`;
                markdown += content + '\n\n';
            }
        });

        markdown += '---\n\n';
        markdown += '## 📊 Summary Statistics\n\n';
        markdown += `- **Original Transcript**: ${(summaryData.original_transcript_words || 0).toLocaleString()} words\n`;
        markdown += `- **Summary Length**: ${(summaryData.word_count || 0).toLocaleString()} words\n`;
        markdown += `- **Processing Time**: ${(summaryData.generation_time_sec || 0).toFixed(1)} seconds\n`;
        markdown += `- **Chunks Processed**: ${summaryData.chunks_successful || 0}/${summaryData.chunks_processed || 0}\n`;

        // Add token usage to markdown
        if (summaryData.token_usage) {
            const tokenUsage = summaryData.token_usage;
            markdown += '\n## 🪙 Token Usage & Costs\n\n';
            markdown += `- **Total Tokens**: ${(tokenUsage.total_tokens || 0).toLocaleString()}\n`;
            markdown += `- **Input Tokens**: ${(tokenUsage.prompt_tokens || 0).toLocaleString()}\n`;
            markdown += `- **Output Tokens**: ${(tokenUsage.completion_tokens || 0).toLocaleString()}\n`;
            markdown += `- **Estimated Cost**: $${(tokenUsage.estimated_cost_usd?.total || 0).toFixed(4)}\n`;

            if (tokenUsage.efficiency_metrics) {
                const efficiency = tokenUsage.efficiency_metrics;
                markdown += `- **Compression Ratio**: ${efficiency.compression_ratio || 0}:1\n`;
                markdown += `- **Processing Speed**: ${(efficiency.processing_speed_words_per_sec || 0).toFixed(0)} words/sec\n`;
                markdown += `- **Efficiency**: ${(efficiency.tokens_per_word_generated || 0).toFixed(1)} tokens per generated word\n`;
            }
        }

        return markdown;
    }

    // Methods for handling visibility without affecting state
    show() {
        console.log('Showing summary component, isReady:', this.isReady);
        this.elements.tabContent.style.display = 'block';
        // Don't change button state when showing - preserve existing readiness
    }

    hide() {
        console.log('Hiding summary component, preserving state');
        // Only hide the display, don't reset the component state or disable buttons
        this.elements.tabContent.style.display = 'none';
    }

    reset() {
        console.log('Resetting summary component');
        this.hide();
        this.currentSummary = null;
        this.isGenerating = false;
        this.isReady = false;
        this.elements.progressContainer.style.display = 'none';
        this.elements.resultContainer.style.display = 'none';
        this.elements.errorContainer.style.display = 'none';
        this.elements.summaryInfo.style.display = 'none';
        // Reset button state
        this.elements.generateBtn.disabled = true;
    }
}
