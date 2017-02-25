所以，创建一个版本库非常简单，首先，选择一个合适的地方，创建一个空目录：
$ mkdir learngit
$ cd learngit
$ pwd
/Users/michael/learngit
pwd命令用于显示当前目录。在我的Mac上，这个仓库位于/Users/michael/learngit。

第二步，通过git init命令把这个目录变成Git可以管理的仓库：
$ git init
Initialized empty Git repository in /Users/michael/learngit/.git/

如果你没有看到.git目录，那是因为这个目录默认是隐藏的，用ls -ah命令就可以看  



#同时上传2个文件
$ git add file1.txt
$ git add file2.txt file3.txt
$ git commit -m "add 3 files."

#常用
$ git status
$ git diff readme.txt 
历史
$ git log
$ git log --pretty=oneline
回到上一个版本就是HEAD^，上上一个版本就是HEAD^^，当然往上100个版本写100个^比较容易数不过来，所以写成HEAD~100。
 git reset --hard HEAD^

1.git init 
2.git config user.name "one" 
3.git config user.email "one@someplace.com" 
4.git add * 
5.git commit -m "msg"

git add . 或者 git add --all
最新的那个版本append GPL已经看不到了！好比你从21世纪坐时光穿梭机来到了19世纪，想再回去已经回不去了，肿么办？
办法其实还是有的，只要上面的命令行窗口还没有被关掉，你就可以顺着往上找啊找啊，找到那个append GPL的commit id是3628164...，于是就可以指定回到未来的某个版本：
$ git reset --hard 3628164
关掉后不知道id了：
$ git reflog

查看内容
$ cat readme.txt
丢弃工作区的修改：
$ git checkout -- readme.txt
add后commit前恢复，丢弃暂存区的修改：：
$ git reset HEAD readme.txt

删除文件：
$ rm test.txt
在你有两个选择，一是确实要从版本库中删除该文件，那就用命令git rm删掉，并且git commit：
$ git rm test.txt
$ git commit -m "remove test.txt"
删错了
$ git checkout -- test.txt
远程
git remote add origin git@github.com:michaelliao/learngit.git
git push origin master  (很重要)
初始化
git bare init--
取消初始化
rm -rf .git

克隆
$ git clone git@github.com:michaelliao/gitskills.git


分支：
查看分支：$ git branch
创建分支并且切换到这里：$ git checkout -b dev   相当于
$ git branch dev  （创建）
$ git checkout dev  （回去）
修改后提交：
$ git add readme.txt 
$ git commit -m "branch test"
回到master分支		$ git checkout master
分支结果合并		$ git merge dev
删除分支：			$ git branch -d dev
两个分支都对文件有修改和提交：打开文件手动删除不需要的,用git log --graph命令可以看到分支合并图。

留下分支的历史不用fast forward方式合并
$ git merge --no-ff   -m "merge with no-ff" dev
$ git log --graph --pretty=oneline --abbrev-commit

有bug建立分支
临时保存未提交的分支：	$ git stash
创建临时分支：			$ git checkout -b issue-101  修改分支，合并结果，删除git branch -d issue-101

查看原来的工作现场： $ git stash list
恢复指定的stash：$ git stash apply stash@{0} 再drop
回到原来的未提交的分支：$git stash pop  相当于
$git stash apply
$git stash drop