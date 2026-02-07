---
layout: default
title: Search
description: Search AI learning articles and tutorials
hero: true
---

<div class="container" style="margin-top: 2rem;">
  <div style="text-align: center; margin-bottom: 3rem;">
    <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Search AI Learning Content</h1>
    <p style="color: #666; font-size: 1.1rem;">Find articles, tutorials, and resources on Artificial Intelligence</p>
  </div>

  <div class="search-box">
    <input type="text" id="search-input" placeholder="Search for articles (e.g., 'neural networks', 'NLP', 'machine learning')...">
  </div>

  <div id="search-results" class="blog-posts-grid" style="display: none;">
    <!-- Search results will appear here -->
  </div>

  <div id="no-results" style="display: none; text-align: center; padding: 3rem;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
    <h3>No results found</h3>
    <p style="color: #666;">Try different keywords or browse our categories</p>
  </div>

  <div id="initial-state" style="text-align: center; padding: 3rem;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">📚</div>
    <h3 style="margin-bottom: 1rem;">Start Your Search</h3>
    <p style="color: #666; max-width: 500px; margin: 0 auto 2rem;">Type in the search box above to find articles on Machine Learning, Deep Learning, NLP, Computer Vision, and more.</p>
    <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
      <span class="filter-btn" onclick="document.getElementById('search-input').value = 'neural networks'; document.getElementById('search-input').dispatchEvent(new Event('input'));">Neural Networks</span>
      <span class="filter-btn" onclick="document.getElementById('search-input').value = 'NLP'; document.getElementById('search-input').dispatchEvent(new Event('input'));">NLP</span>
      <span class="filter-btn" onclick="document.getElementById('search-input').value = 'computer vision'; document.getElementById('search-input').dispatchEvent(new Event('input'));">Computer Vision</span>
      <span class="filter-btn" onclick="document.getElementById('search-input').value = 'generative ai'; document.getElementById('search-input').dispatchEvent(new Event('input'));">Generative AI</span>
    </div>
  </div>
</div>

<script>
// Store all posts for searching
const posts = [
  {% for post in site.posts %}
  {
    title: {{ post.title | jsonify }},
    url: "{{ post.url | relative_url }}",
    excerpt: {{ post.excerpt | strip_html | jsonify }},
    date: "{{ post.date | date: "%B %-d, %Y" }}",
    categories: [{% for category in post.categories %}"{{ category }}",{% endfor %}],
    pathType: "{{ post.path_type | default: '' }}",
    readTime: "{{ post.read_time | default: '' }}"
  },
  {% endfor %}
];

const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
const noResults = document.getElementById('no-results');
const initialState = document.getElementById('initial-state');

searchInput.addEventListener('input', function() {
  const query = this.value.toLowerCase().trim();

  if (query === '') {
    searchResults.style.display = 'none';
    noResults.style.display = 'none';
    initialState.style.display = 'block';
    return;
  }

  initialState.style.display = 'none';

  const filteredPosts = posts.filter(post => {
    return post.title.toLowerCase().includes(query) ||
           post.excerpt.toLowerCase().includes(query) ||
           post.categories.some(cat => cat.toLowerCase().includes(query));
  });

  if (filteredPosts.length === 0) {
    searchResults.style.display = 'none';
    noResults.style.display = 'block';
  } else {
    noResults.style.display = 'none';
    searchResults.style.display = 'grid';
    renderResults(filteredPosts);
  }
});

function renderResults(results) {
  searchResults.innerHTML = results.map(post => `
    <article class="blog-post-card">
      <div class="post-header">
        <span class="post-date">${post.date}</span>
        ${post.pathType ? `<span class="post-path-badge path-badge-${post.pathType.toLowerCase()}">${post.pathType}</span>` : ''}
      </div>
      <h2 class="post-title">
        <a href="${post.url}">${post.title}</a>
      </h2>
      <p class="post-excerpt">${post.excerpt.substring(0, 160)}...</p>
      <div class="post-footer">
        <div class="post-categories">
          ${post.categories.slice(0, 3).map(cat => `<span class="category-tag">${cat}</span>`).join('')}
        </div>
        ${post.readTime ? `<span class="read-time">${post.readTime} min read</span>` : ''}
      </div>
    </article>
  `).join('');
}
</script>