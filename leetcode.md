## Array

48

Array

```java
class Solution {
  //常规 创建一个新的数组 来用保存反转
    public void rotate(int[][] matrix) {
        int n = matrix.length;
      //只有square 才可以被旋转
      //如果是一个长方形，那么形状不满足，无法in order 旋转， new 一个负责我也不知道干嘛
        int[][] matrix_new = new int[n][n];
      
        for (int i = 0; i < n; ++i) {
          //ROW 从 1 开始
            for (int j = 0; j < n; ++j) {
              // fuck ++i & ++j -> no difference with i++ & j++
              //所以就是一个简单的保存换位置
              //然后就是直接简单重新复制一遍就可以了
              //0 4, 1 4, 2 4, 3 4, 4 4
              //0 3, 1 3, 2 3, 3 3, 3 4
              // i 负责row的位置	在简单的--基础上，递增来保存新的数组
              // j 负责col的位置， 就是简单的--
              // 逻辑简单，给的题解很恶心
                matrix_new[j][n - i - 1] = matrix[i][j];
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix[i][j] = matrix_new[i][j];
            }
        }
    }
}
```

```java
class Solution {
  //
    public void rotate(int[][] matrix) {
        int n = matrix.length; // matrix length
      
        for (int i = 0; i < n / 2; ++i) {
          // n + 1 范围
            for (int j = 0; j < (n + 1) / 2; ++j) {
              
                int temp = matrix[i][j];

                matrix[i][j] = matrix[n - j - 1][i];

                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];

                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }
}
// 1. 首先保存开始位置的数值
// 2. 把他要换的数值保存在 开始位置
// 3. 把 他要还的位置数值 保存在2位置
// 4. 完成正常一个数字的对换

// 实现过程
// 1. i j -> (n - j - 1, i)
// 2. (n - j - 1, i) -> (n - i - 1, n - j - 1)
// 3. (n - i - 1, n - j - 1) -> (j, n - i - 1)
// 4. (j, n - i - 1) = temp
```

```java
class Solution {
  //水平翻转 主对角线翻转， 找规律的大成
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        // 水平翻转
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - i - 1][j];
                matrix[n - i - 1][j] = temp;
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }
}

```



54

Array

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> order = new ArrayList<Integer>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return order;
        }
        int rows = matrix.length, columns = matrix[0].length;
        boolean[][] visited = new boolean[rows][columns];
        int total = rows * columns;
        int row = 0, column = 0;
      //这个顺序有没有什么说法
      //第一次改变column+1
      //第二次row+1
      //第三次column-1
      //第四次row-1
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int directionIndex = 0;
        for (int i = 0; i < total; i++) {
          // 首先直接添加
            order.add(matrix[row][column]);
          //设定当前位置为访问过
            visited[row][column] = true;
          //row 是【】【】中的第一个
          //column 是【】【】中的第二个
          //一开始directionindex为0就是一个个方向
            int nextRow = row + directions[directionIndex][0], nextColumn = column + directions[directionIndex][1];
          //如果当前row或者column变为异常的时候，
          //directionindex + 1 % 4
          // row += 当前index的方向
          // 如果下一个位置visitied过了，那么直接改变方向
            if (nextRow < 0 || nextRow >= rows || nextColumn < 0 || nextColumn >= columns || visited[nextRow][nextColumn]) {
                directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            column += directions[directionIndex][1];
        }
        return order;
    }
}

```

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> order = new ArrayList<Integer>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return order;
        }
        int rows = matrix.length, columns = matrix[0].length;
      //right 就是一行最后位置的保存
      // bottom就是 一列最后位置的保存
        int left = 0, right = columns - 1, top = 0, bottom = rows - 1;
      // 四个方向是要通过++ -- 来不断的改变位置
      // 首先left 不能超过right， 顶部不能小于底部
        while (left <= right && top <= bottom) {
          // loop
          // 首先是递归col
        	// 直接在top也就是当前，通过col++的方法，来添加
          // top不变
          // col++
            for (int column = left; column <= right; column++) {
                order.add(matrix[top][column]);
            }
          //loop
          // row = top + 1，下一行， row 小于bottom，++
          // right不变
          // row++
            for (int row = top + 1; row <= bottom; row++) {
                order.add(matrix[row][right]);
            }
          
          //以上步骤完成左->下
          
          // 如果 left < right, top < bottom
          	// 内部loop循环
          	// right开始，bottom不变，col--
            if (left < right && top < bottom) {
                for (int column = right - 1; column > left; column--) {
                    order.add(matrix[bottom][column]);
                }
              // bottom开始，left不限，row--
                for (int row = bottom; row > top; row--) {
                    order.add(matrix[row][left]);
                }
            }
          
            left++;
            right--;
            top++;
            bottom--;
        }
        return order;
    }
}
```



268 finding missing number

Arrays

```java
class Solution {
    public int missingNumber(int[] nums) {
        int sum = 0;
        for (int i = 1; i <= nums.length; i++) {
            sum += i;
            sum -= nums[i - 1];
        }
        return sum;
    }
}

// 数学运算
```

```java
class Solution {
    public int missingNumber(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for (int num : nums) set.add(num);


        for (int i = 0; i <= nums.length; i++) {
            if (!set.contains(i)) {
                return i;
            }
        }
        return -1;
    }
}

// set来计算
```



## Backtracking

784

backtracking

```java
 List<String> res = new ArrayList<>();
    public List<String> letterCasePermutation(String S) {
        char[] chs = S.toCharArray();
        int n = chs.length;
        dfs(chs, n, 0);
        return res;
    }

    private void dfs(char[] chs, int n, int begin) {
        res.add(new String(chs));
        for(int i = begin; i < n; i++){
            if(!Character.isDigit(chs[i])){
                char tmp = chs[i];
                chs[i] = (char)(chs[i] - 'a' >= 0 ? chs[i] - 32 : chs[i] + 32);
                dfs(chs, n, i + 1);
                chs[i] = tmp;
            }
        }
    }
```



78

backtracking

```java
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    public List<List<Integer>> subsets(int[] nums) {

        List<Integer> path = new ArrayList<Integer>();
        dfs(nums, 0, nums.length, path);
        return res;
    }

    void dfs(int[] nums, int begin, int n, List<Integer> path) {
      // 这条也是直接添加的
        res.add(new ArrayList<>(path));

        for (int i = begin; i < n; i++) {
            path.add(nums[i]);
            dfs(nums, i+1, n, path);
          // remove就是直接remove这个list size的最后一个 很经典
            path.remove(path.size() - 1);
        }
    }
}
```



90

backtracking

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<Integer> path = new ArrayList<>();
      // 首先要sort 方面剔除重复元素
        Arrays.sort(nums);
        dfs(nums, 0, nums.length, path);
        return res;
     }
    
    void dfs(int[] nums, int begin, int n, List<Integer> path) {
        res.add(new ArrayList<>(path));
        
        for (int i = begin; i < n; i++) {
          // 去重
					// i != begin, make sure that i - 1 != -1, cause i will = 0
            if (i != begin && nums[i] == nums[i - 1]) {
                continue;
            }
            path.add(nums[i]);
            dfs(nums, i+1, n, path);
            path.remove(path.size() - 1);
            
        }
    }
}
```



46 Permutation

backtracking

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        List<Integer> path = new ArrayList<>();
        for (int num : nums) {
            path.add(num);
        }
        dfs(nums, 0, path);
        return res;
    }

    void dfs(int[] nums, int begin, List<Integer> path) {
        if (nums.length == begin) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = begin; i < nums.length; i++) {
            Collections.swap(path, i, begin);
            dfs(nums, begin + 1, path);
            Collections.swap(path, i, begin);
        }
    }

    void swap(List<Integer> path, int first, int second) {
        int temp = path.get(first);
        path.set(first, path.get(second));
        path.set(second, temp);
        
    }
}
```

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        dfs(nums, 0, nums.length);
        return res;
    }

    void dfs(int[] nums, int first, int n) {
        if (first == n) {
            List<Integer> output = new ArrayList<>();
            for (int i : nums) {
                output.add(i);
            }
            res.add(output);
        }
        for (int i = first; i < n; i++) {
            swap(nums, i, first);
            dfs(nums, first + 1, n);
            swap(nums, i, first);
        }
    }

    void swap(int[] nums, int first, int second) {
        int temp = nums[first];
        nums[first] = nums[second];
        nums[second] = temp;
    }
}
```



47 permutations II

backtracking

```java
class Solution {
    boolean[] vis;

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> perm = new ArrayList<Integer>();
      // boolean 来看看这个的数字是否被拜访过
        vis = new boolean[nums.length];
      // sort一哈 避免重复 1 2 1这种情况出现？
        Arrays.sort(nums);
      // dfs
        backtrack(nums, ans, 0, perm);
        return ans;
    }

    public void backtrack(int[] nums, List<List<Integer>> ans, int idx, List<Integer> perm) {
        if (idx == nums.length) {
            ans.add(new ArrayList<Integer>(perm));
            return;
        }
      // 
        for (int i = 0; i < nums.length; ++i) {
          	
          // 如果当前访问过 直接跳
          // 如果 i i - 1 一样， 并且i - 1也访问过了， 那么跳过
          // 如果 i i - 1 一样， 并且i - 1 没有访问过， 那么继续
          // 这保证了没有重复
            if (vis[i] || (i > 0 && nums[i] == nums[i - 1] && vis[i - 1])) {
                continue;
            }
          // add
            perm.add(nums[i]);
          // 当前index -> true
            vis[i] = true;
          // backtrack
            backtrack(nums, ans, idx + 1, perm);
          // 重置
            vis[i] = false;
          // 重置
            perm.remove(idx);
        }
    }
}

```

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> permuteUnique(int[] nums) {
        dfs(nums, 0);
        return res;
    }

    void dfs(int[] nums, int index) {
        if (index == nums.length) {
            List<Integer> output = new ArrayList<>();
            for (int i : nums) {
                output.add(i);
            }
            res.add(output);
        }

        Set<Integer> set = new HashSet<>();
        for (int i = index; i < nums.length; i++) {
            swap(nums,i, index);
            if (set.add(nums[index])) {
                dfs(nums, index + 1);
            }
            swap(nums,i, index);
        }
    }

    void swap(int[] nums, int first, int second) {
        int temp = nums[first];
        nums[first] = nums[second];
        nums[second] = temp;
    }
}
```



77  combinations

backtracking

```java
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (k <= 0 || n < k) {
            return res;
        } 
        Deque<Integer> path = new ArrayDeque<>();
        dfs(n, k, 1, path, res);
        return res;
    }

    void dfs(int n, int k, int begin, Deque<Integer> path, List<List<Integer>> res) {
        if (path.size() == k) {
            res.add(new ArrayList<>(path));
            return;
        }
         for (int i = begin; i <= n; i++) {
            // 向路径变量里添加一个数
            path.addLast(i);
            // 下一轮搜索，设置的搜索起点要加 1，因为组合数理不允许出现重复的元素
            dfs(n, k, i + 1, path, res);
            // 重点理解这里：深度优先遍历有回头的过程，因此递归之前做了什么，递归之后需要做相同操作的逆向操作
            path.removeLast();
        }
    }
}
```



39 combination sum

Backtracking

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Deque<Integer> path = new ArrayDeque<>();
        dfs(candidates, target, 0, path);
        return res;
    }

    void dfs(int[] candidates, int target, int begin, Deque<Integer> path) {
        if (target < 0) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList(path));
            return;
        } 
    
        for (int i = begin; i < candidates.length; i++) {
            path.addLast(candidates[i]);
            dfs(candidates, target - candidates[i], i, path);
            path.removeLast();
        }
    }
}
```



40 combination sum II

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Deque<Integer> path = new ArrayDeque<>();
        Arrays.sort(candidates);
        dfs(candidates, target, 0, path);
        return res;
    }

    void dfs(int[] candidates, int target, int begin, Deque<Integer> path) {
        if (target < 0) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList(path));
            return;
        } 
    
        for (int i = begin; i < candidates.length; i++) {
          // i > begin 很重要 不是>0
            if (i > begin && candidates[i] == candidates[i - 1]) {
                continue;
            }
            path.addLast(candidates[i]);
            dfs(candidates, target - candidates[i], i + 1, path);
            path.removeLast();
        }
    }
}
```

216 combination sum III

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> combinationSum3(int k, int n) {
        Deque<Integer> path = new ArrayDeque<>();
        dfs(n, k, 1, path);
        return res;
    }

    void dfs(int n, int k, int begin, Deque<Integer> path) {
        if (n == 0 && k == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
    
        for (int i = begin; i <= 9; i++) {
        
            path.addLast(i);
            dfs(n-i, k -1, i + 1, path);
            path.removeLast();
        }
    }
}
```



22 generating parentheses

backtracking

```java
class Solution {
    List<String> res = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        dfs("", n , n);
        return res;
    }
    
    void dfs(String output, int left, int right) {
        if (left == 0 && right == 0) {
            res.add(output);
            return;
        }
        
        if (left > 0) {
            dfs(output + "(", left - 1, right);
        }
        if (right > left) {
            dfs(output + ")", left, right - 1);
        }
    }
}
```



494

```java
class Solution {
    int count = 0;
    public int findTargetSumWays(int[] nums, int target) {
        dfs(nums, target, 0, 0);
        return count;
    }
    
    void dfs(int[] nums, int target, int output, int begin) {
        if (begin == nums.length) {
            if (target == output) {
                count++;               
            } 
            return;
        }
        // 创建一个index指针来keep tracking
        dfs(nums, target, output + nums[begin], begin + 1);
        dfs(nums, target, output - nums[begin], begin + 1);
    }
}
```





## Two pointers

125 valid Palindrome

```java
class Solution {
    public boolean isPalindrome(String s) {
        int n = s.length();
        int left = 0 , right = n - 1;
        while (left < right) {
          // 如果left是空白 直接跳过
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                left++;
            } 
          // 如果right是空白 直接跳过
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                right--;
            } 
          //判断
            if (left < right) {
                if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                    return false;
                }
            }
            left++;
            right--;
        }
        return true;
    }
}
```



## Graph Traversal

94 Binary tree inorder traversal 中序

Inorder 首先遍历左子树，然后访问根结点，最后遍历右子树

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        traversal(res, root);
        return res;
    }

    void traversal(List<Integer> res, TreeNode root) {
        if (root == null) {
            return;
        }
      // recurse 到最左边，null 返回， 添加当前root.val 
        traversal(res, root.left);
        res.add(root.val);
        traversal(res, root.right);
    }
}
```



100 same tree

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        if (p.val != q.val) {
            return false;
        } else {
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }
    }
}
```



133 clone graph



```java
class Solution {
    private HashMap <Node, Node> visited = new HashMap <> ();
    public Node cloneGraph(Node node) {
        if (node == null) {
            return node;
        }

        // 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
        if (visited.containsKey(node)) {
            return visited.get(node);
        }

        // 克隆节点，注意到为了深拷贝我们不会克隆它的邻居的列表
        Node cloneNode = new Node(node.val, new ArrayList());
        // 哈希表存储
        visited.put(node, cloneNode);

        // 遍历该节点的邻居并更新克隆节点的邻居列表
        for (Node neighbor: node.neighbors) {
            cloneNode.neighbors.add(cloneGraph(neighbor));
        }
        return cloneNode;
    }
}

// time complexity: O(N) 	需要遍历所有node的值
// space complexity: O(N) 需要保存node和CloneNode的值

// DFS
// 首先是用visited 来保存cloneNode
// 如果当前node已经被访问过了，防止无限循环，所以需要直接在vistied图中直接返回当前node的cloneNode
// 创建当前node节点的cloneNode, 并且把node -> cloneNode保存在visited中
// 遍历node，使cloneNode获取其neighbors
```

```java
class Solution {
    public Node cloneGraph(Node node) {
        if (node == null) {
            return node;
        }

        HashMap<Node, Node> visited = new HashMap();

        // 将题目给定的节点添加到队列
        LinkedList<Node> queue = new LinkedList<Node> ();
        queue.add(node);
        // 克隆第一个节点并存储到哈希表中
        visited.put(node, new Node(node.val, new ArrayList()));

        // 广度优先搜索
        while (!queue.isEmpty()) {
            // 取出队列的头节点
            Node n = queue.remove();
            // 遍历该节点的邻居
            for (Node neighbor: n.neighbors) {
                if (!visited.containsKey(neighbor)) {
                    // 如果没有被访问过，就克隆并存储在哈希表中
                    visited.put(neighbor, new Node(neighbor.val, new ArrayList()));
                    // 将邻居节点加入队列中
                    queue.add(neighbor);
                }
                // 更新当前节点的邻居列表
                visited.get(n).neighbors.add(visited.get(neighbor));
            }
        }

        return visited.get(node);
    }
}

// BFS
```





## Fast slow pointers



141 linked list cycle

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) return false;

        ListNode fast = head.next, slow = head;
        while (fast != slow) {
            if (fast == null || fast.next == null) return false;
            fast = fast.next.next;
            slow = slow.next;
        }
        return true;
    }
}

// fast 指针指向head.next 因为while loop的循坏条件是fast != slow
// 如果fast || fast.next == null， 停止loop， 答案必定为false
```



876 middle of the linked list

```java
class Solution {
    public ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        return slow;
    }
}
// 快慢指针
```



234 Palindrome linked list

```java
class Solution {
    public boolean isPalindrome(ListNode head) {
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode temp = head;
        slow = reverse(slow);
        while (slow != null) {
            if (temp.val != slow.val) {
                return false;
            }
            slow = slow.next;
            temp = temp.next;
        }
        return true;
    }
    
    ListNode reverse(ListNode head) {
        ListNode curNode = null;
        while (head != null) {
            ListNode temp = head.next;
            head.next = curNode;
            curNode = head;
            head = temp;
        }
        return curNode;
    }
}

// 先取中间段的node
// 然后reverse一下中间断
// 通过一个while loop 比较reverse中间段和head
```



203 remove linked list 

```java
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode dummy = new ListNode(-1, head);
        ListNode temp = dummy;
        while (temp != null && temp.next != null) {
            if (temp.next.val == val) {
                temp.next = temp.next.next;
            } else {
                temp = temp.next;
            }
        }
        return dummy.next;
    }
}

// if next.val = val, remvoe
// else temp = temp.next
```



83 remove duplicates from sorted array

```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {

        ListNode temp = head;
        while (temp != null && temp.next != null) {
            if (temp.next.val == temp.val) {
                temp.next = temp.next.next;
            } else {
                temp = temp.next;
            }
        }
        return head;
    }
}
```



## DFS

111 minimum depth of binary tree

```java
class Solution {
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else if (root.right == null && root.left == null) {
            return 1;
        } 

        if (root.left == null) return minDepth(root.right) + 1;
        if (root.right == null) return minDepth(root.left) + 1;
        return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
    }
}
```



112 path sum

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        if (root.right == null && root.left == null && root.val == targetSum) {
            return true;
        }
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }
}
```



543 diameter of binary tree

```java
class Solution {
    int max = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        dfs(root);
        return max;
    }
    int dfs(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = dfs(root.left);
        int right = dfs(root.right);
        max = Math.max(left + right, max);
        return Math.max(left, right) + 1;
    }
}

// 注意不是经过根节点的最长路径
// 是这棵树中最长路径
// 所以需要一个max来保存right + left的最大值
```



# Note

Recursion / Backtracking 

a. 39. Combination Sum [https://leetcode.com/problems/combina...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmlZcXdhNXZCSktyZWZKQnZXYnNjM3BZUE5GUXxBQ3Jtc0trZV9BT21kdnRoMzNJX0J4ZEdMUUdodDNzcEtrMms4WmtrUWVTZ0gtVmxPakF6Uk5PNVhRa1V0Q1g4b1FsMlhpTkY0UC1vRF9mOVZtbFpPc1lOVVJpcWpnRjF4SHEyMDJOTVQ5MUllaWhfX0dTcWZfVQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fcombination-sum%2F) 

b. 40. Combination Sum II [https://leetcode.com/problems/combina...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbm96VE1nenVuZFhzNzBKSzRIODVnc3ZieFdvZ3xBQ3Jtc0trMlMxWk9nOUNpemFfQmpHb1VGQUlDX3dnRWZ5V1RhMVNGV19UUlc2ai15QWM3XzlCaVhsbC1PQlB1WndWaXl6SzVLSUtMM05pMzcxRC01TUd0ZGJScWN3MTJqd2J5RXJFUjVKSWI4TVgzQzdpQmg0cw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fcombination-sum-ii%2F) 

c. 78. Subsets [https://leetcode.com/problems/subsets/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazFNc0hKcWJCUWdnZVlEZDVtTWVWOHE0Umt1QXxBQ3Jtc0tuOWp3MUpZRnp6U1hIUExoQjR4WW9DLWNHZldJMkduZUF6Tzl0RHROdzQtOV91eWJhQnVHSzdlc3FTdG9FVDBNTmlTZFNKUUhELXV0dmdtUHJCRFdjMWZrUGpjYUdkYWVVd2JJWmJfTGdSaGRGZUVNMA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fsubsets%2F) d. 90. Subsets II [https://leetcode.com/problems/subsets...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmZNLWFBSkdyRXJfQ2pySGtlcGhZR3ZZZEdHZ3xBQ3Jtc0trZEVBUjFFamZIZHp2eGJMampERE5wV2JnMUpTUm9qbHJHeGFfU3Vic3haR1N2Xzg2QklfNkVrY0VoN2ZDRDBjUFJta1ZjMy1pTzNlOVM0VG1IN0xfVUZBQzBHMW9ueWp3ci1Sd2p5OXVCSlEtektUOA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fsubsets-ii%2F) 

e. 46. Permutations [https://leetcode.com/problems/permuta...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbTBHdUo4TnJXMTViVTRZZWJwRExnMFQwaS1kZ3xBQ3Jtc0tsTndINU9laGxhU0ZrNnhPRUtpdldUTXRIbXd1SU5kdnJ5UjZFSGw4LVRuMEFNUnFIQjRzMl95QW9yWklmekxha1RFbjltQjd3V2FEbUw2UmJBdW5EQTNiUTkwSWJ2UDhwZWFHUVlPbk1NejMxWHhDdw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fpermutations%2F) 

f. 47. Permutations II [https://leetcode.com/problems/permuta...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbDlld25zYWY3V2lyLTc1Z1JXNnRSdk81LVlTd3xBQ3Jtc0ttTWtOU09Sak9NdUREZHhOSXg2NVF4Rk5CWVA2VHhHTW4zTEdyd2tmek9oZDJMd0YwQ20wSUloSnhob20xeGpYcVBDNnJkMGprRG1XUi1oaHRTNUFheWJVaS1yRkppeHdVcHA4NXBXV09uTlBwZnZoYw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fpermutations-ii%2F) 



Graph Traversal - DFS, BFS, Topological Sorting

 a. 133. Clone Graph [https://leetcode.com/problems/clone-g...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbHROeFNsaTR5c0VHc3VYSGkwRzhhVlBQSjJoQXxBQ3Jtc0tsWHBCMDZCbW1maXNuRnduU1o4TjJkRHlHdzVvdTZ4QUxuWDE0Tmx5OWlnX1lRS2RMX2VLei1uRmtVay00Ym9YODM1VUxmV3pHcmlZdThmYWNoM0tSWjctVnZUYk50bUhqU3c1Zm01c1FnamxNaXJYMA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fclone-graph%2F) BFS / DFS 

b. 127. Word Ladder [https://leetcode.com/problems/word-la...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa2VrOGFPT1dadng3cHV1VU9TLWF4VVFJODlsQXxBQ3Jtc0ttSnFtamZDYUhmc1hGcjY4N3lTQTR5LTR5QkxISHcyNFdfV0VpSjhBWjRNd2tuLTBjYVV0ZmJOVC1HRXJwakpUc0oxa3IwQXcwdS1rbGRuc0I4aGtRbURQbGZPSmcxUWhvMVFCdEhDbG9oU3g2a2psVQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fword-ladder%2F) BFS 

c. 490. The Maze [https://leetcode.com/problems/the-maze/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbk1LVWxnUDV4RjR5YThhTE9rd1ZiY1dDdHBnQXxBQ3Jtc0tsT3JNV2pBdklvczlxakdCYzFrWWh3RlZoM1JOVk5ESWZ6ZXhtXzNHM1Z0TFZLWVpMQ0x1YmJGT0lKVHY3V3M3MVhGNTVUelBkWXdaYVdEb1N3LVMwbHFoalNmZDdzM005dFBGcE5zT1NWNzdqMExrMA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fthe-maze%2F)  

d. 210. Course Schedule II [https://leetcode.com/problems/course-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa28yT2Nua0V4dE54ek5NRkZscmZUQWJYUm9qUXxBQ3Jtc0tuVzJlX0g0Y2plUzBBVndUNXlERGFvdGhyVV9rUHZQcUtWSmFuZHZDeW4zWjEwZEtMSkItamN3cGJkMXlLb0t4MlZqazF4c2RZNDB5RW4tVmNWb205b3RHemtVTFA4SVdNWWpDWlVtT21ycFQ0U0U4aw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fcourse-schedule-ii%2F) (Topological Sorting) 

e. 269. Alien Dictionary [https://leetcode.com/problems/alien-d...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGg2UWpfdkk3bUMtMnNuTDZlSlpvcElXUzB1d3xBQ3Jtc0tuZHBSN2RmRTJyOERncmRQc3lfQVA2ODdXZGJNWkRNdC1pMkQ3SjMzaEFiRFNUcVhHRE1UN1JVRk80SVVhMEF3NzRISFoxdkVNaWpIMVNuUU13dFNTNGFKT09pZW55YmtSZ0pzYVlQYXRTMEtxLTZrbw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Falien-dictionary%2F) (Topological Sorting) 



Binary Tree / Binary Search Tree (BST)

 a. 94. Binary Tree Inorder Traversal [https://leetcode.com/problems/binary-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbUh6RzlLenBVdENGZnF6MTNfQXFyVXJ5SkpPQXxBQ3Jtc0ttV29jWWtMUTJ6aFJBZlN6OUJ6dXY3TVd2eXBkYmZJbVlnTlFyZmk1Sk5TVzh4Um92d1YyZG1UZXBUV29rQktHenNGR0pWS21BODRrZ1NNS0lLZkJlMlZ2NjlfWnhITVp0NHRZR1hUSkpRZS1SV1Vvcw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fbinary-tree-inorder-traversal%2F)

 b. 236. Lowest Common Ancestor of a Binary Tree [https://leetcode.com/problems/lowest-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbjJmV25wcnJLbGNBVjU5aGoyTFVMMGFYaWVPZ3xBQ3Jtc0ttTG1mdXI1MFV6Xzh2YXFLX1p6RFlUMi1xaWJheFNaNHk5azlqeXFYd2RKQXQwaGxiVFdHdGtueEYxa3R2MFRQR1cwTEdPaDhVeXFxSGZTenVtZUFSY2M0TFR0QUZ6Sko0UU1RNkltUk5QX0s0Y1dLdw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Flowest-common-ancestor-of-a-binary-tree%2F) 

c. 297. Serialize and Deserialize Binary Tree [https://leetcode.com/problems/seriali...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbUw4VVU2bVdzN1lTRHNnbGg3VkZRUXhINGsxQXxBQ3Jtc0trVEJkX1lfWHA5Q3hnemVMLUltSC12Zks5RlNCamc2M3lIYUpCOE1DOExGSHBJdGQxSjZrSWNVM0ljeU0xTFJpQ0h1MHNhUG83M1BSa0ZqRXd5alVwRlEyek5Lb2pYR3kzYzdHRlRGMzdHQW5FazJ1MA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fserialize-and-deserialize-binary-tree%2F) 

d. 98. Validate Binary Search Tree [https://leetcode.com/problems/validat...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa2c5Wm1uV3dWWVR6X0pWVTMydEpSQWFETlZVQXxBQ3Jtc0ttYTZnUU5WR2FuY2dJNERBSmlFdEFqbHFyYUFtc3pCWXpJVFR4UUx3UU1nMmhLYTd1ZkQ0bjZWUmtqYzI3WnFINm9TSGZqZXlWOUM3MnRXRVZMbkNWU24xZUN3emJockhOM2JfTjdSd29PdUZJVEFzWQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fvalidate-binary-search-tree%2F) 

e. 102. Binary Tree Level Order Traversal [https://leetcode.com/problems/binary-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazRhZXFlbFdfcHVzMHJSUkFuRlg2cUdaYUE5QXxBQ3Jtc0trc2YwYmY0Q1VHYmpKRURYWGlBOHhHMFdxS01kTDRNcGcxS1A1VDJWS0xRM2dMME52SmtxZUtNYmcxREp2ZzZGQnNlM0RNd1RNQWpjNU15ZnpaSnlydEVfdEJsMkhGSndfNW5ESXhwd1JycVFNTVdvQQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fbinary-tree-level-order-traversal%2F) (BFS) 



Binary Search

 a. 34. Find First and Last Position of Element in Sorted Array [https://leetcode.com/problems/find-fi...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbF9Sem4zaG01VmJpLWgyVXRGWjctVTIwV0F4UXxBQ3Jtc0tuZ0dXenhjb2JpV2R5NkllaFZoZUhKRnRpZ0xPcDMxOGZFZ2cxREhVUVgzWlhPNHZlY1FXOVZKRnNJLU1xTXRDVU1Qb2VBOGRTOXExTUZ3Ykp5RnN4aWlfOS1LMlBVWEd1MTBZZ3ZaakJnMUw0NU1MTQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Ffind-first-and-last-position-of-element-in-sorted-array%2F) 

b. 162. Find Peak Element [https://leetcode.com/problems/find-pe...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblZiVzFWaXFQUGxKRGs0azZ2djN0OEEzRDllQXxBQ3Jtc0tuWm1GVzZ5MlNENElta3luQ2xkblJPbXZNVWZPbHhVdWQxNDBMT2s2OURHc3puaERTVGMzdkE4R045RnRKcGhtb1dLVmp4b0o3bXBuc1JtUTAyMVlENld1MkJ5eUFHU2p5OVl2cF9YZ3ZtNW0xN3g0MA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Ffind-peak-element%2F)

 c. 69. Sqrt(x) [https://leetcode.com/problems/sqrtx/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa2psbUlpdzdFaURGb0FaSFI1TXhfaF8zcGlxd3xBQ3Jtc0trMF9zOXhMTTZhN0hwVUVlWU1pQ2tHRmpwY2lSSm1uR1gydEpsenlobjhWRkZoVlhqUXFxS2VhUjA1NmEtUHdESXRVT0k1UUdXaXVVQzVwUUJrOTAzVl9KSElsOEdVN3Q4SWpkYmZWSmRGSERtZXExVQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fsqrtx%2F)   



Data Structure

 a. 242. Valid Anagram [https://leetcode.com/problems/valid-a...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbDF1OTBZX1ZqclVqLTh3Q2NtV2RSaUYwMmNzZ3xBQ3Jtc0trUFp0WnkzQ0hxaWoyd21LVHVQbEVkT1ZrdDRpdEkzTkhwWm5LZTdMbGx2MVYxaFRrWklYMkNrMi1VSFZtdXRpYVdMUEkta0U5NXY2d1U3cFpSSVpFdnY1WVNtUThHOGNsbUxYblNWbXJrMkxpOEkwOA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fvalid-anagram%2F) (Hash Table) 

b. 133. Clone Graph [https://leetcode.com/problems/clone-g...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbGpidXJnRGJKcGRCQ0ZtNXpHV3NYVEZkTG9pQXxBQ3Jtc0ttWDN3U3BZTVZFNmxQZ1FFWXJnbWl5LUE5eEp5MUs1cS1hX2NTTFNUR21JYWpDWldjVndfdjZUVDd6VXg3ay1mdDZGdDFPY0ZpOTFUblJvVUFQVk83SUlrNmZtSEV6M0N1UndCNUJOcWZReWFzR3Vjaw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fclone-graph%2F) (Hash Table) 

c. 127. Word Ladder [https://leetcode.com/problems/word-la...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa19veGN3ZXdpeWs4bTlVMnZ4VUhJaENUS0pxUXxBQ3Jtc0tsMGRnSXFLeVBiQUtjSzc2VUVNZjNnTXc2VFlYMW5JSmlGS3k5cXE4ak5FNjVzNkZ3a1VENHpXa2NKQmZoeC1GZVB4T3hwUkZhTlRNNHlOcjZQRWtHRTM1MmhIeEpsUEpGakVIWkJMbnRKeW16MUE0UQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fword-ladder%2F) (Hash Table) 

d. 155. Min Stack [https://leetcode.com/problems/min-stack/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbk9FN1ZEZ014Y0tfY0NSeERENnZRSkFqQndfd3xBQ3Jtc0tsVG1RRDFMc3BKcHJ3anphVEV3ZTdLbUJocl9BV0xVaGtBeGNRZUlydDk1blB2UGJVbllMU0VKMkRGTkxDRTRaa0I2bTVmRDNFZlJLZ0dubkc4VktHNFhKNnVPMEFQUVdxRXE5QWlfclN0aXJoWV81Zw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fmin-stack%2F) (Stack) 

e. 225. Implement Stack using Queues [https://leetcode.com/problems/impleme...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbmJiaGtrTlpvQ05JZGhZODJsM2t2UkpLTGZHUXxBQ3Jtc0tsb3pfcnJ1NC0zQUxrWlNMTlpUM1liV2RMODEyalZSa1ViVkQtRHZsUG9WdFhBMEJScEZ0QVZWX1ExSHdZS0pMbGpFZEZXQlVBcEJkMTJPU0lDOU5CajM4VWNhZVRkdWJudDZRaTdsMEhVMllZdTZETQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fimplement-stack-using-queues%2F) (Stack / Queue)

 f. 215. Kth Largest Element in an Array [https://leetcode.com/problems/kth-lar...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa0NERXR3c0p6ZzFKNnNjak53cmdhdF9Rak5wQXxBQ3Jtc0tua0k2aWZPb1JuSFlubmxEX3NiQ0g2WlZjUG5rLWMtRm5hZ3RRZGZOLWYyTWJqazNnZUxlTTVzcjFTR3FEQTBXMl9XS1pRdjgzOXBMUjVzTTQwYWE3WEhnUV9zaXN5ZE1mN2pWNGRPZjRLbFhoZW9mOA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fkth-largest-element-in-an-array%2F) (PriorityQueue) 

g. 23. Merge k Sorted Lists [https://leetcode.com/problems/merge-k...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqblFlbjVfeGhaT2l6UkRET0FrZHFYWjA0SU80QXxBQ3Jtc0tsRHE1aHduVWV6TjVBSWc1WGU0Y29zUHZmcXJST3I4aVBzdDZweWlyeHhwdVRXSW92anFiVjM0SUVMek93M2Y1MktMdDUzUGdnNVc3aUJZTVpIMXN2WmZuc2REQjFydjh3aFpfQmYwdE9wUWZvOF8wNA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fmerge-k-sorted-lists%2F) (PriorityQueue) 



Linked List Manipulation 

a. 237. Delete Node in a Linked List [https://leetcode.com/problems/delete-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbk9sek9BU0wxMUZRTG1HZUlQcmVQNVZsenEyUXxBQ3Jtc0tuaG56clZIR2pIeDg3M0dBbU9oekM1RUN4djhtem04M2IwdDhWVlUyX3pkUjZtMGM2YVAtTFNYS1lnUmkwbDNlN1ppU2R1MDd0dDFQMWUyTXZSajg2OE9zZDJWcUF0Z09ScGNGcTRuc1FEZXN5YWZxSQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fdelete-node-in-a-linked-list%2F) 

b. 92. Reverse Linked List II [https://leetcode.com/problems/reverse...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbHZyWVpwczV2R2dJTzBhVE9va0YzeVFxbVVmd3xBQ3Jtc0trbWdybkRjSG1MeENqMDRtb21Zd2FMVWVrVkQ1dnlDOGV4cFdfSFFaNnZYYjJfWGlHVWxpdklUX0RDTU9PWXFWU2tQVjFicjRRdHp3d1lQd3NCeDVhUWk1VDd5NDB4R3NsY1hkQ2dYSFJDa0J6d09TQQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Freverse-linked-list-ii%2F) 

c. 876. Middle of the Linked List [https://leetcode.com/problems/middle-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXpETzdGOGFGeTlZaWMtR0ExaXVIWjVfYUZnQXxBQ3Jtc0ttUzJEcUVlb1FjNjhwS0l4OU5ad1M2eFVpS2dWcmNxVG5ocUJCNmFxbzBtblJXekJaWDJSdm9LOXR5dlVvSnRRTVNtd0RYQjdwWnE4ZUtKYlFCRXBGSzgzTzAxeXFGNE02Vld6QWl5aHNGVFkyTXdqTQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fmiddle-of-the-linked-list%2F) 

d. 143. Reorder List [https://leetcode.com/problems/reorder...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbjBIZWVIX2tTV0NNMWFoY1RuWlJrektQamk5QXxBQ3Jtc0ttZXdoSDMwTzczSmRBeUxHRmMtR3R3eDgycUdGQlhuQWVkN3BYNzQ5WFVZMWNEWU52TlFnUDk4RTFmMVRwaHVWZGE5VV9Gb09adzZCYnEwSEZzeU9qWTFaVUQ5b3dtUEZWQjFVTGFhUmNRelppWHBMRQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Freorder-list%2F)



Pointer Manipulation 

a. 239. Sliding Window Maximum [https://leetcode.com/problems/sliding...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXY1d2Itc3NDcHl2Yklxd2E3SWlaa19uZXdUUXxBQ3Jtc0tuTmY3WmFScjhTR1lzX3Fncm96cDFNUHNOWmR6Y1M5Z3pOSXh6WDBZQ0lldTJNM0g4UUtpRGhyLWdhLU02OVUwamc2WmlJSTdPaGZMMDl1RzBsVExhVndTZ3U5eXJjam94clFpRkRaZEhKa2l0UGREWQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fsliding-window-maximum%2F) 

b. 3. Longest Substring Without Repeating Characters [https://leetcode.com/problems/longest...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFNFbmlJUVpuaDJfZXItbjhmMmRscTNERnFGQXxBQ3Jtc0ttRGNNeERraVoxVnhyMEZKdDR0ZlI1U2t3X3R1cm9jWURBU0FPR3o1ME5Ob2wzWDNOSE43ZjRZbDlzbHdNd1ZVQVhXQ2YxTFBmYUVOdU9ST25XMVh2Qm55eUFBSWUtMEZLVFVUcEc0LVF4NHY4WmlXZw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Flongest-substring-without-repeating-characters%2F) 

c. 76. Minimum Window Substring [https://leetcode.com/problems/minimum...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbE1FVjFCb0VyYWppSERfcTNzeWp4ZkVOTkxpQXxBQ3Jtc0trTjBSOE9GZjIyLUo4NXBMQWNzaDlreHNHUE9uVmJ3eTNxbnBuMG1VWHFoaU5DZXp5WXB1cUFRUEcyZVd6c0hkY1Axb1loa09jeG9qY19QRGw2RFcwM0R4RVBFT2I3Y3lNVFN3SVZ5aURYTkZTSnNsOA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fminimum-window-substring%2F) 



Sorting 

a. Time -- O(N log N) 

b. Merge Sort -- Space O(N) 

c. Quick Sort 

d. 148. Sort List [https://leetcode.com/problems/sort-list/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbHM3bUJOSXhIOGJJQ3h5MWpXZFlwbU1XUjhMd3xBQ3Jtc0tteG5vQUtxYkVVbVdQbS1iU2VBX1JOSFBzeTctN215OW0xenI4aHMyZ3hENFJMQzdkc1ZYUW5VM0sydWdSX2hweHY0a1BsT3pfQlRTam9fYXRGU2czZGtjclA0MUFlZzh0TTl2VHhsUTV0X3dKZTdrNA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fsort-list%2F) 



Convert Real Life Problem to Code  

a. 146. LRU Cache [https://leetcode.com/problems/lru-cache/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbFJHNklvU3k1blhFcW5fMDI3SUJSQU5ZTERtQXxBQ3Jtc0trWGtFTmhGTkQ1R3VfMDVLaThZRk1CbHNBZ0VPb2tWdlJlWlFsbDdFWWwwOTZrZVpiN191X2ZRaHVtYUd5b051Y0pYcUVhWVNhNG1PS1Nhd1lCMFJVc1czbU15cDJ4ZGJRUHhEaGtDYzNucjE4QzVMNA&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Flru-cache%2F) 

b. 1066. Compus Bike [https://leetcode.com/problems/campus-...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbDZuc0Q4QnVpb2JLLWd1bl9EMXNHU3ZjMlhlZ3xBQ3Jtc0tsQjU2Mko5U3ptaFB0NWdhTGRmc214c291bEVBRlZvb0VESEVJcDk4cmtGV1pycDhNaFJQTG1hVUdlb0J6QnJqdkt3Y1huZ3VDbTZVVlVxWTdVZnJQZGxTdHNpS1J5QjM3MmJyWGlMbGRDVXRWWU8wZw&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fcampus-bikes%2F) 

c. 490. The Maze [https://leetcode.com/problems/the-maze/](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbVFwbTNRVXMydy1oWkhRSFhmalZGZFlySURHUXxBQ3Jtc0ttaVRUZFhYN0JRRENPYkhkNWdqR1JxajhIVTJWRldKSVl6bEg0RVRMcUlKd1lzY0ZlbGpIMEFqdnJ6V3ZBVUtUUmp6R0FKX1J1OWFoRE0wZ1dWVkk5UnUxdnlUamlnV3RZVVlNeW9PYWp4MFl5bU4xTQ&q=https%3A%2F%2Fleetcode.com%2Fproblems%2Fthe-maze%2F)  



Time Space Complexity 

a. 一般面试的时候 你说完算法 就要说 这个算法的 time / space complexity是什么

 b. 每次你做完一道题 给自己养成一个习惯 就是想一下他的时间空间复杂度是多少
