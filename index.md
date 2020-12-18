<table border="0">
  <tr>
    <td width="75%">
      <h1>蒋凯涛</h1>
      <p><b>硕士研究生</b></p>
      <p><b>西安交通大学人工智能学院</b></p>
      <p><b>邮箱：1143958845@qq.com</b></p>
    </td>
    <td width="25%">
      <img src="/imgs/github2.jpg" width="100%">
    </td>
  </tr>
</table>

### 研究方向 计算机视觉 深度学习

### 最新文章
#### 参考我的[CSDN](https://mp.csdn.net/console/column/allColumnList)博客
#### [leetcode](blogs/leetcode/test.md)
#### [computer vision](blogs/cv/test.md)



<!-- Gitalk 评论 start  -->
{% if site.gitalk.enable %}
<!-- Gitalk link  -->
<link rel="stylesheet" href="https://unpkg.com/gitalk/dist/gitalk.css">
<script src="https://unpkg.com/gitalk@latest/dist/gitalk.min.js"></script>

<div id="gitalk-container"></div>
    <script type="text/javascript">
    var gitalk = new Gitalk({
    clientID: '{{site.gitalk.clientID}}',
    clientSecret: '{{site.gitalk.clientSecret}}',
    repo: '{{site.gitalk.repo}}',
    owner: '{{site.gitalk.owner}}',
    admin: ['{{site.gitalk.admin}}'],
    distractionFreeMode: {{site.gitalk.distractionFreeMode}},
    id: 'resources',
    });
    gitalk.render('gitalk-container');
</script>
{% endif %}
<!-- Gitalk end -->

 <!-- disqus 评论框 start  -->
{% if site.disqus.enable %}

<div class="comment">
    <div id="disqus_thread" class="disqus-thread">
    </div>
</div>
<!-- disqus 评论框 end -->

<!-- disqus 公共JS代码 start (一个网页只需插入一次) -->
<script type="text/javascript">
    /* * * CONFIGURATION VARIABLES * * */
    var disqus_shortname = "{{site.disqus.username}}";
    var disqus_identifier = "{{site.disqus.username}}/{{page.url}}";
    var disqus_url = "{{site.url}}{{page.url}}";

    (function() {
        var dsq = document.createElement('script'); dsq.type = 'text/javascript'; dsq.async = true;
        dsq.src = '//' + disqus_shortname + '.disqus.com/embed.js';
        (document.getElementsByTagName('head')[0] || document.getElementsByTagName('body')[0]).appendChild(dsq);
    })();
</script>
<!-- disqus 公共JS代码 end -->
{% endif %}
