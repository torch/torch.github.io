---
layout: default
title: Blog
id: blog
---

## Blog Posts

{% for post in site.posts %}
  <li>
    <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
	<a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
	<br>
	{{ post.excerpt }}
  </li>
{% endfor %}
