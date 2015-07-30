---
layout: default
title: Blog
id: blog
---

{% include nav_blog.html %}

{% for post in site.posts %}
### [ {{ post.title }} ]({{ post.url }})
{% include post_detail.html %}

{% endfor %}

{% include footer.html %}
