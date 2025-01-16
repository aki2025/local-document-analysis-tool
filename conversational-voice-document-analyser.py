from nltk import word_tokenize, pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
import spacy
from datetime import datetime
import re

class DocumentContext:
    """Maintains context of the current document interaction session."""
    def __init__(self):
        self.current_document = None
        self.current_page = None
        self.last_query = None
        self.conversation_history = []
        self.relevant_sections = []
        
    def update_context(self, query, response):
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now()
        })

class NaturalLanguageProcessor:
    """Processes natural language queries and extracts intent and entities."""
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        
    def extract_intent(self, query):
        """Extract the user's intent from the query."""
        doc = self.nlp(query.lower())
        
        # Common intent patterns
        intents = {
            'find': r'(find|search|look for|show|where is)',
            'explain': r'(explain|tell me about|what is|how does)',
            'compare': r'(compare|difference between|vs|versus)',
            'summarize': r'(summarize|summary|brief|overview)',
            'navigate': r'(go to|open|jump to|navigate)',
            'filter': r'(filter|only show|limit to)'
        }
        
        for intent, pattern in intents.items():
            if re.search(pattern, query.lower()):
                return intent
                
        return 'general'
    
    def extract_entities(self, query):
        """Extract relevant entities from the query."""
        doc = self.nlp(query)
        entities = {
            'dates': [],
            'numbers': [],
            'keywords': [],
            'topics': [],
            'filters': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
            elif ent.label_ == 'CARDINAL':
                entities['numbers'].append(ent.text)
            elif ent.label_ in ['ORG', 'PERSON', 'GPE']:
                entities['keywords'].append(ent.text)
        
        # Extract filter conditions
        filter_patterns = {
            'date_range': r'(from|between)\s+(.*?)\s+(to|and)\s+(.*?)(?:\s|$)',
            'category': r'in\s+(\w+)',
            'type': r'type\s+of\s+(\w+)'
        }
        
        for filter_type, pattern in filter_patterns.items():
            matches = re.findall(pattern, query)
            if matches:
                entities['filters'].append({
                    'type': filter_type,
                    'value': matches
                })
        
        return entities

class DocumentFilter:
    """Filters and ranks documents based on natural language queries."""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
    
    def apply_filters(self, documents, entities):
        """Apply extracted filters to documents."""
        filtered_docs = documents
        
        for filter_info in entities.get('filters', []):
            if filter_info['type'] == 'date_range':
                filtered_docs = self._filter_by_date_range(
                    filtered_docs, 
                    filter_info['value']
                )
            elif filter_info['type'] == 'category':
                filtered_docs = self._filter_by_category(
                    filtered_docs, 
                    filter_info['value']
                )
        
        return filtered_docs
    
    def _filter_by_date_range(self, documents, date_range):
        """Filter documents within a date range."""
        start_date, end_date = self._parse_date_range(date_range)
        return [
            doc for doc in documents
            if self._is_in_date_range(doc, start_date, end_date)
        ]
    
    def _filter_by_category(self, documents, category):
        """Filter documents by category or type."""
        return [
            doc for doc in documents
            if self._matches_category(doc, category)
        ]
    
    def rank_results(self, documents, query, top_k=5):
        """Rank filtered documents by relevance to query."""
        query_doc = self.nlp(query)
        
        scored_docs = [
            (doc, self._calculate_relevance(doc, query_doc))
            for doc in documents
        ]
        
        return sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]

class EnhancedVoiceInterface(VoiceInterface):
    """Enhanced voice interface with natural conversation capabilities."""
    
    def __init__(self):
        super().__init__()
        self.nlp = NaturalLanguageProcessor()
        self.doc_filter = DocumentFilter()
        self.context = DocumentContext()
    
    async def process_voice_query(self, audio_input):
        """Process voice input with natural language understanding."""
        # Convert speech to text
        query = self.speech_to_text(audio_input)
        
        # Extract intent and entities
        intent = self.nlp.extract_intent(query)
        entities = self.nlp.extract_entities(query)
        
        # Process based on intent
        response = await self._handle_intent(intent, entities, query)
        
        # Update context
        self.context.update_context(query, response)
        
        return response
    
    async def _handle_intent(self, intent, entities, query):
        """Handle different types of user intents."""
        if intent == 'find':
            return await self._handle_find_intent(entities, query)
        elif intent == 'explain':
            return await self._handle_explain_intent(entities, query)
        elif intent == 'filter':
            return await self._handle_filter_intent(entities, query)
        elif intent == 'navigate':
            return await self._handle_navigation_intent(entities, query)
        else:
            return await self._handle_general_query(query)
    
    async def _handle_find_intent(self, entities, query):
        """Handle queries about finding specific information."""
        filtered_docs = self.doc_filter.apply_filters(
            self.documents,
            entities
        )
        
        ranked_results = self.doc_filter.rank_results(filtered_docs, query)
        
        response = self._format_find_response(ranked_results)
        await self._navigate_to_result(ranked_results[0][0])
        
        return response
    
    async def _handle_filter_intent(self, entities, query):
        """Handle document filtering requests."""
        filtered_docs = self.doc_filter.apply_filters(
            self.documents,
            entities
        )
        
        response = self._format_filter_response(filtered_docs, entities)
        self.context.relevant_sections = filtered_docs
        
        return response
    
    def _format_find_response(self, ranked_results):
        """Format the response for find queries."""
        if not ranked_results:
            return "I couldn't find any relevant information."
        
        response = "I found the following relevant information:\n\n"
        for doc, score in ranked_results:
            response += f"- {doc.summary}\n"
            response += f"  (Found in {doc.metadata['file_path']})\n\n"
        
        return response
    
    def _format_filter_response(self, filtered_docs, entities):
        """Format the response for filter queries."""
        response = f"I've filtered the documents based on your criteria. "
        response += f"Found {len(filtered_docs)} relevant documents.\n\n"
        
        for doc in filtered_docs[:3]:  # Show top 3 as examples
            response += f"- {doc.summary}\n"
        
        if len(filtered_docs) > 3:
            response += f"\nAnd {len(filtered_docs) - 3} more results..."
        
        return response

# Update the Flask routes to handle enhanced voice interactions
@app.route('/process_voice_query', methods=['POST'])
async def process_voice_query():
    """Handle voice queries with natural language understanding."""
    audio_data = request.files['audio'].read()
    
    try:
        response = await voice_interface.process_voice_query(audio_data)
        return jsonify({
            'status': 'success',
            'response': response,
            'context': {
                'current_document': voice_interface.context.current_document,
                'relevant_sections': len(voice_interface.context.relevant_sections)
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Add this to the HTML template
"""
<script>
    // Add natural language query handling
    async function handleVoiceQuery(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob);
        
        const response = await fetch('/process_voice_query', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (result.status === 'success') {
            // Update UI with response
            updateResponseArea(result.response);
            
            // Update document view if navigation occurred
            if (result.context.current_document) {
                updateDocumentView(result.context.current_document);
            }
            
            // Speak response
            await speakResponse(result.response);
        }
    }
    
    function updateResponseArea(response) {
        const responseArea = document.getElementById('response-area');
        responseArea.innerHTML = `
            <div class="response-content">
                ${response}
            </div>
        `;
    }
    
    function updateDocumentView(document) {
        const documentView = document.getElementById('document-view');
        documentView.innerHTML = `
            <div class="document-content">
                ${document.content}
            </div>
        `;
    }
</script>
"""
