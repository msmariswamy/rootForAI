---
layout: default
title: Topics
description: Browse AI topics by category
hero: true
---

<div class="topics-grid" id="topics">
  <h2 style="margin-bottom: 1.5rem;">Explore Topics</h2>
  {% for topic in site.topics %}
  <a href="{{ '/rootForAI/topics/' | relative_url }}{{ topic.id }}" class="topic-card" style="border-left-color: {{ topic.color }};">
    <h3 style="color: {{ topic.color }};">{{ topic.title }}</h3>
    <p>View all {{ topic.title }} articles</p>
  </a>
  {% endfor %}
</div>
