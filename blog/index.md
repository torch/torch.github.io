---
layout: default
title: Blog
id: blog
---

{% for post in site.posts %}
### [ {{ post.title }} ]({{ post.url }})
{% include post_detail.html %}

{% endfor %}
