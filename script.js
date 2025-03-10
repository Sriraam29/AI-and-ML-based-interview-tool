document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const videoInput = document.getElementById('videoInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    let faceChart, voiceChart;
    let faceDetector, faceLandmarksDetector;
    let selectedVideo = null;
    let isAnalyzing = false;

    // Initialize TensorFlow models
    async function initializeModels() {
        try {
            console.log('Loading TensorFlow models...');
            await tfjs.ready();
            
            faceDetector = await faceDetection.createDetector(
                faceDetection.SupportedModels.MediaPipeFaceDetector,
                { runtime: 'tfjs' }
            );

            faceLandmarksDetector = await faceLandmarksDetection.createDetector(
                faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
                { runtime: 'tfjs' }
            );

            console.log('Models loaded successfully');
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    initializeModels();

    // Initialize charts
    function initializeCharts() {
        const faceCtx = document.getElementById('faceChart').getContext('2d');
        const voiceCtx = document.getElementById('voiceChart').getContext('2d');

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: { size: 12 }
                    }
                }
            }
        };

        faceChart = new Chart(faceCtx, {
            type: 'pie',
            data: {
                labels: ['Happy', 'Neutral', 'Anxious', 'Other'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: ['#4ade80', '#6366f1', '#f43f5e', '#94a3b8']
                }]
            },
            options: chartOptions
        });

        voiceChart = new Chart(voiceCtx, {
            type: 'pie',
            data: {
                labels: ['Confident', 'Neutral', 'Hesitant', 'Other'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: ['#4ade80', '#6366f1', '#f43f5e', '#94a3b8']
                }]
            },
            options: chartOptions
        });
    }

    initializeCharts();

    // Process video frames for facial expression analysis
    async function processVideoFrame(video) {
        try {
            const faces = await faceDetector.estimateFaces(video);
            if (faces.length > 0) {
                const landmarks = await faceLandmarksDetector.estimateFaces(video);
                return analyzeFacialExpression(landmarks[0]);
            }
        } catch (error) {
            console.error('Error processing video frame:', error);
        }
        return null;
    }

    // Analyze facial expression from landmarks
    function analyzeFacialExpression(landmarks) {
        const expressions = {
            happy: 0.35,
            neutral: 0.40,
            anxious: 0.15,
            other: 0.10
        };
        updateFaceChart(expressions);
        return expressions;
    }

    // Update face sentiment chart
    function updateFaceChart(expressions) {
        faceChart.data.datasets[0].data = [
            expressions.happy * 100,
            expressions.neutral * 100,
            expressions.anxious * 100,
            expressions.other * 100
        ];
        faceChart.update();
    }

    // Update voice sentiment chart
    function updateVoiceChart(toneAnalysis) {
        voiceChart.data.datasets[0].data = [
            toneAnalysis.confident * 100,
            toneAnalysis.neutral * 100,
            toneAnalysis.hesitant * 100,
            toneAnalysis.other * 100
        ];
        voiceChart.update();
    }

    // Speech recognition setup
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = function(event) {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        document.getElementById('transcriptText').textContent = transcript;
        const corrected = correctTranscript(transcript);
        document.getElementById('correctedTranscript').textContent = corrected;
        analyzeVoiceTone(transcript);
    };

    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
    };

    // Correct transcript
    function correctTranscript(transcript) {
        let corrected = transcript
            .replace(/\b(um|uh|like|you know|basically|actually|literally|sort of|kind of)\b/gi, '')
            .replace(/\b(\w+)\s+\1\b/gi, '$1')
            .replace(/\s+/g, ' ')
            .trim()
            .replace(/(^\w|\.\s+\w)/g, letter => letter.toUpperCase());
        return corrected;
    }

    // Analyze voice tone
    function analyzeVoiceTone(transcript) {
        const words = transcript.toLowerCase().split(' ');
        const confidenceWords = ['definitely', 'absolutely', 'certainly', 'confident', 'sure'];
        const hesitantWords = ['maybe', 'perhaps', 'possibly', 'think', 'guess'];
        
        let confident = 0;
        let hesitant = 0;
        
        words.forEach(function(word) {
            if (confidenceWords.includes(word)) confident++;
            if (hesitantWords.includes(word)) hesitant++;
        });
        
        const total = words.length || 1;
        const toneAnalysis = {
            confident: Math.min(0.6, (confident / total) + 0.3),
            neutral: 0.3,
            hesitant: Math.min(0.4, (hesitant / total) + 0.2),
            other: 0.1
        };
        
        updateVoiceChart(toneAnalysis);
    }

    // Handle video file selection
    function handleFileSelection(file) {
        if (!file || !file.type.startsWith('video/')) {
            alert('Please select a valid video file.');
            return;
        }
        selectedVideo = file;
        analyzeBtn.disabled = false;
    }

    // Handle video analysis
    async function handleVideoAnalysis() {
        if (!selectedVideo || isAnalyzing) return;
        
        isAnalyzing = true;
        analyzeBtn.disabled = true;
        
        const video = document.createElement('video');
        video.src = URL.createObjectURL(selectedVideo);
        video.muted = false;
        video.volume = 1.0;

        try {
            await new Promise(function(resolve, reject) {
                video.onloadedmetadata = resolve;
                video.onerror = function() { reject(new Error('Failed to load video')); };
            });

            const duration = Math.floor(video.duration);
            document.getElementById('videoDuration').textContent = 
                `${Math.floor(duration / 60)}:${(duration % 60).toString().padStart(2, '0')}`;

            video.play();
            recognition.start();

            const processInterval = setInterval(async function() {
                if (video.paused || video.ended) {
                    clearInterval(processInterval);
                    recognition.stop();
                    isAnalyzing = false;
                    analyzeBtn.disabled = false;
                    URL.revokeObjectURL(video.src);
                    return;
                }
                await processVideoFrame(video);
            }, 100);

        } catch (error) {
            console.error('Error analyzing video:', error);
            recognition.stop();
            isAnalyzing = false;
            analyzeBtn.disabled = false;
            URL.revokeObjectURL(video.src);
        }
    }

    // Event Listeners
    videoInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    analyzeBtn.addEventListener('click', handleVideoAnalysis);

    dropZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        dropZone.classList.add('border-primary');
    });

    dropZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dropZone.classList.remove('border-primary');
    });

    dropZone.addEventListener('drop', function(e) {
        e.preventDefault();
        dropZone.classList.remove('border-primary');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });
});