from flask import Flask, request, jsonify
from retrieval import retrieve_candidates, generate_response

app = Flask(__name__)

@app.route('/match', methods=['POST'])
def match():
    job_description = request.json['job_description']
    retrieved_candidates = retrieve_candidates(job_description)
    responses = generate_response(retrieved_candidates)
    return jsonify(responses)

if __name__ == '__main__':
    app.run(debug=True)
