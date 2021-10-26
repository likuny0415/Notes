## Array

41

```java
    public int firstMissingPositive(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && nums[nums[i]-1] != nums[i]) {
                swap(nums, i, nums[i]-1);
            }
        }
        
        for(int i = 0; i < nums.length; i++) {
            if (nums[i] != i+1)    return i+1;
        }
        return nums.length+1;
    }
    
    private void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

// 主要是将数字调整到该换位置然后 loop一下找对对应出问题的地方
```





128 longest consecutive sequence

medium

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        
        for (int n : nums) {
            set.add(n);
        }
        
        int max = 0;
        for (int num : nums) {
            if (!set.contains(num - 1)) {
                int curNum = num;
                int curLongest = 1;
                
                while (set.contains(curNum + 1)) {
                    curLongest++;
                    curNum++;
                }
                
                max = Math.max(max, curLongest);
            }
        }
        return max;
    }
}

// 网上查，不往下查

// O(n)
// O(n)
```





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



54 spiral matrix

medium

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
        List<Integer> res = new ArrayList<>();

        if (matrix.length == 0 || matrix[0].length == 0 || matrix == null) {
            return res;
        }

        // length & width
        int n = matrix.length, m = matrix[0].length;
        int left = 0, right = m - 1, top = 0, bottom = n - 1;

        while (left <= right && top <= bottom) {
            // add first column to the list
            for (int i = left; i <= right; i++) {
                res.add(matrix[top][i]);
            }

            // add right row to the list
            for (int i = top + 1; i <= bottom; i++) {
                res.add(matrix[i][right]);
            }

            if (left < right && top < bottom) {
                // add bottom to the list
                for (int i = right - 1; i > left; i--) {
                    res.add(matrix[bottom][i]);
                } 

                // add left to the list
                for (int i = bottom; i > top; i--) {
                    res.add(matrix[i][left]);
                }
            }
            left++;
            right--;
            top++;
            bottom--;
        }
        return res;
    }
}

// 1. 注意loop的条件
		// add top (Col, [不变][]) -> for (int i = left; i <= right; i++)
																// res.add(matrix[top][i]);
		// add right (Row [][不变]) -> for (int i = top + 1; i <= bottom; i++)
																// res.add(matrix[i][right]);
		// add bottom (Col, [不变][]) -> for (int i = right - 1; i > left; i--)
																// res.add(matrix[bottom][i]);
		// add left (Row [][不变]) -> for (int i = bottom - 1; i > top; i--)
																// res.add(matrix[i][left]);
// O(m n)
// O(1)
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



### Insert Intevals

57

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> res = new ArrayList<>();

        int first = newInterval[0];
        int second = newInterval[1];

        int i = 0;
        while (i < intervals.length && intervals[i][1] < first) {
            res.add(intervals[i]);
            i++;
        }

        while (i < intervals.length && intervals[i][0] <= second) {
            first = Math.min(intervals[i][0], first);
            second = Math.max(intervals[i][1], second);
            i++;
        }
        res.add(new int[]{first, second});

        while (i < intervals.length) {
            res.add(intervals[i]);
            i++;
        }
        return res.toArray(new int[0][]);
    }
}

// 通过三段法来判断边界，第一段没有任何关系，第二段开始处理关系，然后结束关系

// O(n)
// O(1)
```





452

```java
class Solution {
    public int findMinArrowShots(int[][] points) {
        if (points.length == 0) {
            return 0;
        }
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] a, int[] b) {
                return a[1] > b[1] ? 1 : -1;
            }
        });
        int arrowPos = points[0][1];
        int arrowCnt = 1;
        for (int i = 1; i < points.length; i++) {
            if (arrowPos >= points[i][0]) {
                continue;
            }
            arrowCnt++;
            arrowPos = points[i][1];
        }
        return arrowCnt;
    }
}
```





56 merge intervals

Medium

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        List<int[]> res = new ArrayList<>();
        int length = intervals.length;
        if (intervals == null || length == 0) return res.toArray(new int[0][]);

        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);

        int i = 0;
        while (i < length) {
            int left = intervals[i][0];
            int right = intervals[i][1];

            while (i < length - 1 && intervals[i + 1][0] <= right) {
                right = Math.max(right, intervals[i + 1][1]);
                i++;
            }
            res.add(new int[]{left, right});
            i++;
        }
        
        return res.toArray(new int[0][]);
    }
}

// 1. sort intervals
// 2. 遍历: 因为left按照顺序,right没有按照顺序,如果下一个left<当前left， merge操作
// 3. merge: while loop 更新right，直到intervals的left> i-1 right
// 4. 添加
// 5. list -> int[][], res.toArray(new int[0][])

// nlog(n)
// log(n)
```







## Two pointers

696

```java
class Solution {
    public int countBinarySubstrings(String s) {
        List<Integer> counts = new ArrayList<Integer>();
        int ptr = 0, n = s.length();
        while (ptr < n) {
            char c = s.charAt(ptr);
            int count = 0;
            while (ptr < n && s.charAt(ptr) == c) {
                ++ptr;
                ++count;
            }
            counts.add(count);
        }
        int ans = 0;
        for (int i = 1; i < counts.size(); ++i) {
            ans += Math.min(counts.get(i), counts.get(i - 1));
        }
        return ans;
    }
}

// O(n)
// O(n)
```

```java
class Solution {
public:
    int countBinarySubstrings(string s) {
        int ptr = 0, n = s.size(), last = 0, ans = 0;
        while (ptr < n) {
            char c = s[ptr];
            int count = 0;
            while (ptr < n && s[ptr] == c) {
                ++ptr;
                ++count;
            }
            ans += min(count, last);
            last = count;
        }
        return ans;
    }
};

// 如果分辨的方法就是，{2, 3, 1, 2}
// 通过俩个相邻和的Min，得出配对结果
// 然后res++

// O(n)
// O(1)
```





658

```java
class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int left = 0, right= arr.length - 1;
        while (right - left >= k) {
            if (Math.abs(arr[left] - x) <= Math.abs(arr[right] - x)) {
                right--;
            } else {
                left++;
            }
        }
        List<Integer> list = new ArrayList<>();
        for (int i = left; i <= right; i++) {
            list.add(arr[i]);
        }
        return list;
    }
}

// 规定范围 然后添加

// O(n)
// O(k)
```





11

```java
class Solution {
    public int maxArea(int[] height) {
        int max = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            int min = Math.min(height[left], height[right]);
            max = Math.max(max, min * (right - left));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return max;
    }
}

// 主要就是一个边界判断的问题

// O(n)
// O(1)

```





75

```java
class Solution {
    public void sortColors(int[] nums) {
        int n = nums.length;
        int p1 = 0, p2 = n - 1;
        for (int i = 0; i < nums.length; i++) {
            while(nums[i] == 2 && i <= p2) {
                swap(nums, i, p2);
                p2--;
            }
            if (nums[i] == 0) {
                swap(nums, i, p1);
                p1++;
            }
        }
    }

    void swap(int[] nums, int i , int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}

// swap && exchange 的规律
// 如果一直是2就一直swap，因为可能遇到i和p2换了之后还是2的状态，知道p2不再是2就没事了
// 如果是0，因为都是从前面换过来的肯定没事，直接走

// O(n)
// O(1)
```





15 

```java
public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(nums);
    for (int i = 0; i + 2 < nums.length; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) {              // skip same result
            continue;
        }
        int j = i + 1, k = nums.length - 1;  
        int target = -nums[i];
        while (j < k) {
            if (nums[j] + nums[k] == target) {
                res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                j++;
                k--;
                while (j < k && nums[j] == nums[j - 1]) j++;  // skip same result
                while (j < k && nums[k] == nums[k + 1]) k--;  // skip same result
            } else if (nums[j] + nums[k] > target) {
                k--;
            } else {
                j++;
            }
        }
    }
    return res;
}
```





88 merge sorted array

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int tail = nums1.length - 1;
        int p1 = m - 1;
        int p2 = n - 1;
        while (p2 >= 0) {
            if (p1 < 0 || nums1[p1] <= nums2[p2]) {
                nums1[tail--] = nums2[p2--];
            } else {
                nums1[tail--] = nums1[p1--];
            }
        }
    }
}

```





15 3 sum

Medium

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }

        int n = nums.length;

        Arrays.sort(nums);
        if (nums[0] > 0) {
            return res;
        }

        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int left = i + 1;
            int right = n - 1;
            int sum = 0;
            while (left < right) {
                sum = nums[left] + nums[i] + nums[right];
                if (sum == 0) {
                    res.add(Arrays.asList(nums[left], nums[i], nums[right]));
                    while (left < right && nums[left] == nums[left+1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right-1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
         return res;
    }
}
```





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



1 two sum

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
              // 反插入
                return new int[]{i, map.get(target - nums[i])};
            }
            map.put(nums[i], i);
        }
        return new int[]{-1,-1};
    } 
}

// time complexity = O(N)
// space complexity = O(1) -> 可以优化，for loop 然后不保存值
```

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i+1; j < nums.length; j++) {
                if (nums[i] + nums[j] == target) {
                    return new int[]{i, j};
                }
            }
        }
        return new int[]{0,0};
    }
}
// 愚蠢版本
```



### Sliding window

239

```java
public int[] maxSlidingWindow(int[] a, int k) {		
		if (a == null || k <= 0) {
			return new int[0];
		}
		int n = a.length;
		int[] r = new int[n-k+1];
		int ri = 0;
		// store index
		Deque<Integer> q = new ArrayDeque<>();
		for (int i = 0; i < a.length; i++) {
			// remove numbers out of range k
			while (!q.isEmpty() && q.peek() < i - k + 1) {
				q.poll();
			}
			// remove smaller numbers in k range as they are useless
			while (!q.isEmpty() && a[q.peekLast()] < a[i]) {
				q.pollLast();
			}
			// q contains index... r contains content
			q.offer(i);
			if (i >= k - 1) {
				r[ri++] = a[q.peek()];
			}
		}
		return r;
	}


// 相当于是保存一个deque用来判断当前数字的大小，只要是小了就去掉，否则的话就相当保留了当前位置的数字，然后不断的遍历来得到正确的树枝



```





424

```java
class Solution {
    public int characterReplacement(String s, int k) {
        int[] map = new int[26];
        int start = 0, maxCount = 0, maxLength = 0;
        
        for (int end = 0; end < s.length(); end++) {
            maxCount = Math.max(maxCount, ++map[s.charAt(end) - 'A']);
            while (end - start + 1 - maxCount > k) {
                map[s.charAt(start++) -'A']--;
            }
            maxLength = Math.max(maxLength, end - start + 1);
        }
        return maxLength;
    }
}
```



567

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int len1 = s1.length(), len2 = s2.length();
        if (len1 > len2) return false;
        
        int[] mapping = new int[26];
        for (int i = 0; i < len1; i++) {
            mapping[s1.charAt(i) - 'a']++;
            mapping[s2.charAt(i) - 'a']--;
        }

        if (allZero(mapping)) return true;
        
        for (int i = len1; i < len2; i++) {
            mapping[(s2.charAt(i) - 'a')]--;
            mapping[(s2.charAt(i- len1) - 'a')]++;
            if (allZero(mapping)) return true;
        }
        return false;
    }
    
    boolean allZero(int[] nums) {
        for (int i = 0; i < 26; i++) {
            if (nums[i] != 0) return false;
        }
        return true;
    }
}

// 第一个loop需要check原因 "a", "ab" -> 这种情况mapping只剩下一个b所以会报错，需要添加进去
// +了一整个len2 又减去一整个(len2 - len1)的部分 = len2 - (len2 - len1) = len1;

```



713

```java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if (k == 0) return 0;
        int count = 0;
        // it is product
        int pro = 1;
        
        for (int i = 0, j = 0; j < nums.length; j++) {
            pro *= nums[j];
            while ( i <= j && pro >= k) {
                pro /= nums[i++];
            }
            count += j - i + 1;
        }
        return count;
    }
}

// two loops is the most easiest way

// O(n)
// O(1)
```





```java
class Solution {
    public int totalFruit(int[] fruits) {
        int j = 0, i = 0;
        int count = 0;
        Map<Integer, Integer> map = new HashMap<>();
        
        while (j < fruits.length) {
            map.put(fruits[j], map.getOrDefault(fruits[j], 0) + 1);
            
            while (map.size() > 2) {
                map.put(fruits[i], map.get(fruits[i]) - 1);
                if (map.get(fruits[i]) == 0) map.remove(fruits[i]);
                i++;
                
            }
            count = Math.max(count, j - i + 1);
            j++;
        }
        return count;
    }
}

// sliding window
// tail删除的条件是重中之重， 这边删除的办法就是通过map一步步减小，直到val == 0,然后移除
// 总数的计算方法就是j - i
```





209 minimum size subarray sum

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int min = Integer.MAX_VALUE;
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            
            for (int j = 0; j < i; j++) {
                if (sum >= target) {
                    sum -= nums[j];
                    min = Math.min(min, i - j);
                }
            }
        }
        return min == Integer.MIN_VALUE ? 0 : min;
    }
}

// 滑动窗口其实主要也是一个条件判断
// 首先两个指针是必不可少的，但是指针可以非常灵活
// 一个指针一定是指向尾巴的，一个指针指向头
// 通过头不断延伸，然后缩短尾巴，在伸出头，在缩短的方式不断向前进
// 然后通过条件来保存所需要的最小条件
// 完成所需要的结果

// O(n)
// O(1)
```





3 longest substring without repeating characters

medium

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int n = s.length();

        int rk = -1, ans = 0;
        for (int i = 0; i < n; i++) {
            if (i != 0) {
                set.remove(s.charAt(i - 1));
            }
            while (rk + 1 < n && !set.contains(s.charAt(rk + 1))) {
                set.add(s.charAt(rk+1));
                rk++;
            }
            ans = Math.max(ans, rk - i + 1);
        }
        return ans;
    }
}

// 1. 像一个窗口一样，不断的往前滑
// 2. rk 相当于左侧窗口的左边，还没有开始滑动
// 3. 不断滑动rk窗口，set添加当前char， 直到当前窗口 = string.length, 或者有重复元素
// 4. 删除i位置的char，继续遍历rk右边的窗口
// 5. 更新ans

// O(n)
// O(∣Σ∣)
```





### Fast Slow Pointer

986 intervals intersections

```java
class Solution {
    public int[][] intervalIntersection(int[][] A, int[][] B) {    
        List<int[]> ans = new ArrayList<>();
        int i = 0, j = 0;

        while (i < A.length && j < B.length) {
            int low = Math.max(A[i][0], B[j][0]);
            int high = Math.min(A[i][1], B[j][1]);
            if (low <= high) {
                ans.add(new int[]{low, high});
            } 
            if (A[i][1] < B[j][1]) {
                i++;
            } else {
                j++;
            }
        }
        return ans.toArray(new int[0][]);
    }
}

// 1. 找到firstList, secondList, left的最大值
// 2. 找到firstList, secondList, right最小值
// 3. 如果left <= right list.add
// 4. i++ -> (first.right < second.right)
// 5. j++ -> (first.right >= second.right)

// O(M + N)
// O(M + N)


```



844  backspace string compare 

Easy

```java
class Solution {
    public boolean backspaceCompare(String S, String T) {
        int i = S.length() - 1, j = T.length() - 1;
        int skipS = 0, skipT = 0;

        while (i >= 0 || j >= 0) {
            while (i >= 0) {
                if (S.charAt(i) == '#') {
                    skipS++;
                    i--;
                } else if (skipS > 0) {
                    skipS--;
                    i--;
                } else {
                    break;
                }
            }
            while (j >= 0) {
                if (T.charAt(j) == '#') {
                    skipT++;
                    j--;
                } else if (skipT > 0) {
                    skipT--;
                    j--;
                } else {
                    break;
                }
            }
            if (i >= 0 && j >= 0) {
                if (S.charAt(i) != T.charAt(j)) {
                    return false;
                }
            } else {
                if (i >= 0 || j >= 0) {
                    return false;
                }
            }
            i--;
            j--;
        }
        return true;
    }
}

// two pointer
// time complexity: O(N + M) 遍历两个字符串
// space complexity: O(1) -> 定义常量的指针和计数器
```



142 Linkedlist cycle II 

Medium

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if (head == null) return null;

        ListNode fast = head, slow = head;
        while (true) {
            if (fast == null || fast.next == null) return null;
            fast = fast.next.next;
            slow = slow.next;
            // when they equal
            if (fast == slow) {
                break;
            }
        }

        fast = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }
}

// why we can't use while (fast != slow) at first place, because it doesn't make sure they meet one step further in the cycle, but somewhere else
// time complexity: O(N)
// space complexity: O(1)
```



2 add two numbers

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode res = new ListNode(-1);
        ListNode temp = res;
        int left = 0, sum, n1, n2;
        while (l1 != null || l2 != null) {
            n1 = l1 == null ? 0 : l1.val;
            n2 = l2 == null ? 0 : l2.val;

            sum = n1 + n2 + left;
            temp.next = new ListNode(sum % 10);
            left = sum / 10;
            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
            temp = temp.next;
        }
        if (left != 0) {
            temp.next = new ListNode(left);
        }
        return res.next;
    }
}
// time: O(MAX(M + N))
// space: O(1)

// 进位操作， left和两个指针的进位都需要注意
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

Times: 2

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

437

```java
class Solution {
    int count = 0;
    public int pathSum(TreeNode root, int targetSum) {
        if (root == null) return count;
        dfs(root, targetSum, 0);
        pathSum(root.left, targetSum);
        pathSum(root.right, targetSum);
        return count;
    }
    void dfs(TreeNode root, int targetSum, int curValue) {
        if (root == null) return;
        curValue += root.val;
        if (curValue == targetSum) count++;
        
        dfs(root.left, targetSum, curValue);
        dfs(root.right, targetSum, curValue);
    }
}

// 可以通过right left subtree 的pathsum来避免搜查漏洞
// O(n2)
// O(n)

【飞猪提醒】01-10 11:25浦东机场T2飞-01-10 20:50爱德华劳伦斯洛根国际机场E降国泰航空CX363/CX812，订单3750983208315(航空公司订单号：211008110818482b446)已成功，LI/KUNYANG 票号160-2311284848。浦东机场起飞前60分钟停止办理值机，登录飞猪查看详情（运输总条件和禁止/限制携带物品等要求国泰航空https://c.tb.cn/F3.0frmnK）。受疫情影响建议您提前安排行程。卖家服务电话：4008886628， 飞猪服务电话：9510208 , 境外用户请拨打 86-0571-56888688。
```





863 all nodes distance k in binary tree

```java
class Solution {
    
    Map<TreeNode, Integer> map = new HashMap<>();
    List<Integer> res = new ArrayList<>();
    
    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {

        find(root, target);
        dfs(root, target, k, map.get(root));
        return res;
    }
    
    private void dfs(TreeNode root, TreeNode target, int k, int length) {
        if (root == null) return;
        if (map.containsKey(root)) length = map.get(root);
        if (length == k) res.add(root.val);
        dfs(root.left, target, k, length  + 1);
        dfs(root.right, target, k, length + 1);
    }
    
    int find(TreeNode root, TreeNode target) {
        if (root == null) return - 1;
        if (root == target) {
            map.put(root, 0);
            return 0;
        }
        int left = find(root.left, target);
        if (left >= 0) {
            map.put(root, left + 1);
            return left + 1;
        }
        int right = find(root.right, target);
        if (right >= 0) {
            map.put(root, right + 1);
            return right + 1;
        }
        return - 1;
    }
}

// dfs里面为什么要添加length, 因为一旦到了target之后，就不会继续走了，所以tree后半部分map里面都是没有的，就只能通过length不断++来判断位置
```





662 maximum width of binary tree

```java
public int widthOfBinaryTree(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        Queue<Integer> qIndex = new LinkedList<>();
        q.add(root);
        qIndex.add(1); //store index, assuming root's index is 1
        int max = 0;
        while(!q.isEmpty())
        {
            int size = q.size();
            int start = 0, end = 0;
            for(int i=0; i<size; i++)
            {
                TreeNode node = q.remove();
                int index = qIndex.remove();
                if(i==0) start = index; //start and end index for each level
                if(i==size-1) end = index;
                if(node.left!=null)
                {
                    q.add(node.left);
                    qIndex.add(2*index);
                }
                
                if(node.right!=null)
                {
                    q.add(node.right);
                    qIndex.add(2*index+1);
                }
            }
            max = Math.max(max, end - start + 1);
        }
        return max;    
    }
```





654 maximum binary tree

```java
class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) { 
        return dfs(nums, 0, nums.length - 1);
    }
    
    TreeNode dfs(int[] nums, int left, int right) {
        if (left > right) return null;
        
        int pos = -1;
        int max = Integer.MIN_VALUE;
        for (int i = left; i <= right; i++) {
            if (nums[i] > max) {
                max = nums[i];
                pos = i;
            }
        }
        TreeNode res = new TreeNode(nums[pos]);
            
        res.left = dfs(nums, left, pos-1);
        res.right = dfs(nums, pos+1, right);
        
    
        return res;
    }   
}

// just build a helper

// O (n 2) because 654321每一次都去找max，所以n * (n -1) * (n-2) -> n2
// O(n) -> save n of elements
```

```java
class Solution {
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        Deque<TreeNode> stack = new LinkedList<>();
        
        for (int i = 0; i < nums.length; i++) {
            TreeNode cur = new TreeNode(nums[i]);
            
            while (!stack.isEmpty() && stack.peek().val < nums[i]) {
                cur.left = stack.pop();
            }
            
            if (!stack.isEmpty()) {
                stack.peek().right = cur;
            }
            stack.push(cur);
        }
        
        return stack.isEmpty() ? null : stack.removeLast();
    }
}

// stack 解决办法
// O(n) average
// O(n)
```







113 path sum II

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        if (root == null) return res;
        dfs(root, targetSum, new ArrayList<>());
        return res;
    }
    
    void dfs(TreeNode root, int sum, List<Integer> path) {
        if (root == null) return;
        path.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            res.add(new ArrayList<>(path));
        } else {
            dfs(root.left, sum - root.val, path);
            dfs(root.right, sum - root.val, path);
        }
        path.remove(path.size() - 1);
    }
}

// 判断条件：left right == null, sum == 0, 添加
// 然后除去之后再判断

// O(n2)
// O(n) : 需要添加n个元素
```





236 lowest common ancestor of a binary tree

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) {
            return right;
        } else if (right == null) {
            return left;
        } else {
            return root;
        }
    }
}
// 1. if root == q || p return
// 2. find left or right
// 3. if left or right both exist, return root
// 4. if left not exist return right
// 5. verse vasa

// O(n)
// O(n)
```



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



235 lowestCommonAncestor

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
      // 先绑定
        TreeNode ans = root;
        while(true) {
          // 如果root.val < q && p
          // root = root.right
            if (ans.val < p.val && ans.val < q.val) {
                ans = ans.right;
          // 如果root.val > q && p
          // root = root.left
            } else if (ans.val > p.val && ans.val > q.val) {
                ans = ans.left;
            // 其他情况说明其中一个大了或者小了，直接return
            } else {
                break;
            }
        }
        return ans;
    }
}
```



572 Subtree of Another Tree

```java
class Solution {
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null) return false;
	
      // 首先判断root==subroot, 然后判断root.left == subroot, root.right == subroot
        return isSameTree(root, subRoot) || isSubtree(root.left, subRoot) || isSubtree(root.right, subRoot);
    }

    boolean isSameTree(TreeNode left, TreeNode right) {
        if (left == null && right == null) return true;
        if (left == null || right == null) return false;
        if (left.val != right.val) return false;
        return isSameTree(left.left, right.left) && isSameTree(left.right, right.right);
    }
}
```



617 merge two binary trees

```java
class Solution {
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return root2;
        }
        if (root2 == null) {
            return root1;
        }
        TreeNode ans = new TreeNode(root1.val + root2.val);
        ans.left = mergeTrees(root1.left, root2.left);
        ans.right = mergeTrees(root1.right, root2.right);
        return ans
    }
}
```



226 invert a binary tree

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return root;

        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
      // 将左边绑定到右边
        root.left = right;
      // 将右边绑定到左边
        root.right = left;

        return root;
    }
}
```



690 employee importance

easy

```java
class Solution {
    Map<Integer, Employee> map = new HashMap<>();
    public int getImportance(List<Employee> employees, int id) {
        for (Employee e: employees) {
            map.put(e.id, e);
        }
        return dfs(id);
    }
    
    int dfs(int id) {
        Employee e = map.get(id);
        int sum = e.importance;
        List<Integer> subs = e.subordinates;
        for (Integer subId : subs) {
            sum += dfs(subId);
        }
        return sum;
    }
}

// hashmap + dfs
// hashmap(id, employee)
// 1. current employee
// 2. get Importance & subordinates
// 3. sum -> dfs(id) & dfs(subs.id)
// 4. sum
// time: O(N)
// space: O(N)
```



200 number of islands

Medium

Times:3

```java
class Solution {
    public int numIslands(char[][] grid) {
        int count = 0;
        int n = grid.length, m = grid[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid,i,j);
                    count++;
                }
            }
        }
        return count;
    }

    void dfs(char[][] grid,int i,int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == '0') {
            return;
        }
        grid[i][j] = '0';
        dfs(grid,i-1,j);
        dfs(grid,i+1,j);
        dfs(grid,i,j-1);
        dfs(grid,i,j+1);
    }
}
// 1.找到 1 也就是岛 count++
// 2. 将岛的部分全部换成1也就是不是岛的部分

// time: O(MN)
// space: O(MN)
```



order list

Medium

```java
class Solution {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode middleNode = findMiddleNode(head);
        ListNode rightSide = middleNode.next;
        middleNode.next = null;

        ListNode left = sortList(head);
        ListNode right = sortList(rightSide);
        return mergeTwoList(left, right);
    }

    ListNode findMiddleNode(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode fast = head.next.next;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    ListNode mergeTwoList(ListNode left, ListNode right) {
        ListNode dummy = new ListNode(-1);
        ListNode temp = dummy;
        while (left != null && right != null) {
            if (left.val < right.val) {
                temp.next = left;
                left = left.next;
            } else {
                temp.next = right;
                right = right.next;
            }
            temp = temp.next;
        }
        temp.next = left == null ? right : left;
        return dummy.next;
    }
}

// fast 一定要多一步 fast = head.next, slow = head
// otherwise -> generate stackoverflow error

// time: nlog(N)
// space: log(N)
```



98 validate binary tree

medium

```java
class Solution {
    public boolean isValidBST(TreeNode root) {
        return dfs(root, null, null);
    }

    boolean dfs(TreeNode root, Integer lower, Integer upper) {
        if (root == null) {
            return true;
        }

        int val = root.val;
        if (lower != null && lower >= val) return false;
        if (upper != null && upper <= val) return false;

        if (!dfs(root.left, lower, val)) return false;
        if (!dfs(root.right, val, upper)) return false;
        return true;
    }
}

// java 语法知识
// 只有integer默认null, int 默认0
// Integer -> object, int -> primary type

// 首先binary tree的规则是所有左边 < 右边， 所以有一个lower upper bound很重要
// 其次就是dfs-> false -> false
// 然后就是lower 和 upper都是被val更新的
// root.left -> upper被限制，lower没有被限制
// root.right -> lower被限制，upper没有限制
```



417 pacific altantic water flow

Times:2

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        int m = heights.length;
        int n = heights[0].length;
        if (heights == null || m == 0 || n == 0) {
            return res;
        }

        boolean[][] canReachP = new boolean[m][n];
        boolean[][] canReachA = new boolean[m][n];

        for (int i = 0; i < n; i++) {
            // y fixed, x change
            dfs(0, i, 0, heights, canReachP); // top
            dfs(m-1, i, 0, heights, canReachA); // bottom
        }

        for (int j = 0; j < m; j++) {
            // x fixed, y change
            dfs(j, 0, 0, heights, canReachP); // left
            dfs(j, n-1, 0, heights, canReachA); // right
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (canReachP[i][j] == true && canReachA[i][j] == true) {
                    List<Integer> validLand = new ArrayList<>();
                    validLand.add(i);
                    validLand.add(j);
                    res.add(validLand);
                }
            }
        }
        return res;
    }


    void dfs(int x, int y, int h, int[][] heights, boolean[][] visited) {
        if (x < 0 || y < 0 || x >= heights.length || y >= heights[0].length) return;
        if (visited[x][y] || heights[x][y] < h) return;
        visited[x][y] = true;
        dfs(x-1, y, heights[x][y], heights, visited);
        dfs(x+1, y, heights[x][y], heights, visited);
        dfs(x, y-1, heights[x][y], heights, visited);
        dfs(x, y+1, heights[x][y], heights, visited);
    }
}

// 1. dfs边界条件要设置清楚
		
    // n = heights.length, m = heights[0].length;
    // n = 宽， m = 长       
		// i < m, j < n;
// 2. top -> y不变，x改变 -> dfs(i, 0)
//    bottom -> y不变，x改变 -> dfs(i, n - 1)
// 3. left -> x不变，y改变 -> dfs(0, j);
//		right -> x不变，y改变 -> dfs(m -1, j);
```



### Topological sort





207 course schedule

```java
class Solution {
    public boolean canFinish(int num, int[][] pre) {
        if (num == 0 || pre == null || pre.length == 0) return true;
        
        List<List<Integer>> courses = new ArrayList<>();
        int[] visited = new int[num];
        
        for (int i = 0; i < num; i++) {
            courses.add(new ArrayList<>());
        }
        
        for (int i = 0; i < pre.length; i++) {
            courses.get(pre[i][0]).add(pre[i][1]);
        }
        
        for (int i = 0; i < num; i++) {
            if (!dfs(i, courses, visited)) return false;
        }
        return true;
    }
    
    boolean dfs(int course, List<List<Integer>> courses, int[] visited) {
        visited[course] = 1;
        
        List<Integer> dependCourses = courses.get(course);
        for (int i = 0; i < dependCourses.size(); i++) {
            int dependCourse = dependCourses.get(i);
            if (visited[dependCourse] == 1) return false;
            if (visited[dependCourse] == 0) {
                if (!dfs(dependCourse, courses, visited)) return false;
            }
        }
        visited[course] = 2;
        return true;
    }
}

// 1. add all depend courses to the current course
// 2. loop through its depend courses
// 3. if there is a loop (has not get out the looop yet!!!) then false;
// 4. no depend and return false.

// O(m + n) m for number of courses, n for prerequisites.length
// O(m + n)
```



210 course schedule 

medium

```java
class Solution {
    public int[] findOrder(int num, int[][] pre) {
        int[] res = new int[num];
        int[] indegree = new int[num];
        int k = 0;
        
        for (int[] pair : pre) {
            indegree[pair[0]]++;
        }
        
        Queue<Integer> q = new LinkedList<>();
        
        for (int i = 0 ; i < indegree.length; i++) {
            if (indegree[i] == 0) {
                res[k++] = i;
                q.offer(i);
            }
        }
        
        while (!q.isEmpty()) {
            int noDependCourse = q.poll();
            
            for (int[] pair : pre) {
                if (pair[1] == noDependCourse) {
                    indegree[pair[0]]--;
                    if (indegree[pair[0]] == 0) {
                        int noMoreDependCourse = pair[0];
                        res[k++] = noMoreDependCourse;
                        q.offer(noMoreDependCourse);
                    }
                }
            }
        }
        return k == num ? res : new int[0];
    }
}

// 1. indegree: course and its number of depend courses
// 2. if no depend course, add to res;
// 3. keep eliminating depend course, and see whether the k == num

// O(m + n) m for numofCourses and n for prerequisite.length
// O(m + n)
```



### Backtracking

51

```java
class Solution {
    List<List<String>> result = new ArrayList<>();
    public List<List<String>> solveNQueens(int n) {
        boolean[] visited = new boolean[n];
        //2*n-1个斜对角线
        boolean[] dia1 = new boolean[2*n-1];
        boolean[] dia2 = new boolean[2*n-1];
        
        fun(n, new ArrayList<String>(),visited,dia1,dia2,0);
        
        return result;
    }
    
    private void fun(int n,List<String> list,boolean[] visited,boolean[] dia1,boolean[] dia2,int rowIndex){
        if(rowIndex == n){
            result.add(new ArrayList<String>(list));
            return;
        }
        
        for(int i=0;i<n;i++){
            //这一行、正对角线、反对角线都不能再放了，如果发现是true，停止本次循环
            if(visited[i] || dia1[rowIndex+i] || dia2[rowIndex-i+n-1])
                continue;
            
            //init一个长度为n的一维数组，里面初始化为'.'
            char[] charArray = new char[n];
            Arrays.fill(charArray,'.');
            
            charArray[i] = 'Q';
            String stringArray = new String(charArray);
            list.add(stringArray);
            visited[i] = true;
            dia1[rowIndex+i] = true;
            dia2[rowIndex-i+n-1] = true;

            fun(n,list,visited,dia1,dia2,rowIndex+1);

            //reset 不影响回溯的下个目标
            list.remove(list.size()-1);
            charArray[i] = '.';
            visited[i] = false;
            dia1[rowIndex+i] = false;
            dia2[rowIndex-i+n-1] = false;
        }
    }
}

// for right diagnoal
col = 0, row = 0, right -> 3 initial(n - 1)
col = 0, row = 1, right -> 4 (+rowIndex)
col = 0, row = 2, right -> 5
col = 1, row = 0, right -> 2 (-i)
col = 2 row = 1, right -> 2
  
// for left diagonal
```





17

```java
class Solution {
    String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

		List<String> res = new ArrayList<>();
    public List<String> letterCombinations(String digits) {
        if (digits.length() == 0) return res;
        dfs(digits,0, "");
        return res;
    }

    void dfs(String digits, int index, String path) {
        if (index == digits.length()) {
            res.add(path);
            return;
        }
        Character c = digits.charAt(index);
        String curList = mapping[c - '0'];
        for (int i = 0; i < curList.length(); i++) {
            dfs(digits, index+1, path + curList.charAt(i));
        }
    }
}

// dfs(每次遍历， 每次+一下还是比较容易理解的)
```

```java
class Solution {
    String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

    public List<String> letterCombinations(String digits) {
        LinkedList<String> list = new LinkedList<>();
        if (digits.length() == 0) return list;
        list.add("");
        for (int i = 0; i < digits.length(); i++) {
            Character c = digits.charAt(i);
            while(list.peek().length() == i) {
                String cur = list.remove();
                for (char curChar : mapping[c - '0'].toCharArray()) {
                    list.add(cur + curChar);
                }
            }
        } 
        return list;
    }
}
```





79 word searching

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        int n = board.length;
        int m = board[0].length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (exist(board, word, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    boolean exist(char[][] board, String word, int x, int y, int index) {
        if (index == word.length()) {
            return true;
        }

        if (x < 0 || y < 0 || x >= board.length || y >= board[0].length) return false;

        if (word.charAt(index) != board[x][y]) return false;

        char c = board[x][y];
        board[x][y] = '0';
        boolean res = exist(board, word, x + 1, y, index + 1) 
        || exist(board, word, x - 1, y, index + 1) 
        || exist(board, word, x, y + 1, index + 1) 
        || exist(board, word, x, y - 1, index + 1);
        board[x][y] = c;
        return res;
    } 
}

// 就是一个边界判断 没有什么特别奇特的地方

// O(mn * 3 ^ L)
// O(mn)
```





131 palindrome partitioning 

```java
class Solution {
    List<List<String>> res = new ArrayList<>();
    
    public List<List<String>> partition(String s) {
        dfs(s, new ArrayList<>(), 0);        
        return res;
    }
    
    void dfs(String s, List<String> path, int begin) {
        if (s.length() == 0) {
            res.add(new ArrayList<>(path));
            return;
        }
        
        for (int i = 0; i < s.length(); i++) {
            if (isPalindrome(s.substring(0,i + 1))) {
                path.add(s.substring(0, i+ 1));
                dfs(s.substring(i+1), path, i + 1);
                path.remove(path.size() - 1);
            }
        }
    }
    
    boolean isPalindrome(String s) {
        for (int i = 0; i < s.length() /2; i++) {
            if (s.charAt(i) != s.charAt(s.length() - 1 - i)) return false;
        }
        return true;
    }
}


// 其实主要做的一个事情就是切割，查看是否是palidrome, 然后继续backtrack
// O(n * 2 ^ n)
// O(n)
```





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

    void dfs(int[] nums, int target, int index, int output) {
        if (index == nums.length) {
            if (target == output) {
                count += 1;
            }
            return;
        }

        dfs(nums, target, index + 1, output + nums[index]);
        dfs(nums, target, index + 1, output - nums[index]);
    }
}

// 只有前进，没有后退，所以index一直++就行，然后的话就是终止条件判断

// O(2^n)
// O(n)
```







## BFS

```java
class Solution {
    public Node connect(Node root) {
        if (root == null) return null; 
        Queue<Node> q = new LinkedList<>();
        q.offer(root);
        while(!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                Node nd = q.poll();
                if (i == size - 1) {
                    nd.next = null;
                } else if (i < size - 1) {
                    nd.next = q.peek();
                }
                if (nd.left != null) {
                    q.offer(nd.left);
                }
                if (nd.right != null) {
                    q.offer(nd.right);
                }
            }
        }
        return root;
    }
}

// O(n)
// O(1)
```







322 coin change

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        Queue<Integer> q = new LinkedList<>();
        q.offer(0);
        
        int res = 0;
        boolean[] visited = new boolean[amount + 1];
        while(!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                int sum = q.poll();
                if (sum == amount) {
                    return res;
                }
                
                
                if (sum > amount || visited[sum]) {
                    continue;
                }
                visited[sum] = true;
                for (int coin : coins) {
                    q.offer(sum + coin);
                }
            }
            res++;
        }
        return -1;
    }
}
```



107 binary tree level order traversal II

```java
class Solution {
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();
        if (root == null) return res;
        q.offer(root);
        
        
        while(!q.isEmpty()) {
            int size = q.size();
            List<Integer> cur = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode temp = q.poll();
                cur.add(temp.val);
                if (temp.left != null) {
                    q.offer(temp.left);
                }
                if (temp.right != null) {
                    q.offer(temp.right);
                }
            }
            res.add(0, cur);
        }
        return res;
    }
}
// add 特殊用法
// O(n)
// O(n)
```





## LinkedList

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(-1, head);
        ListNode preNd = dummy;

        for (int i = 0; i < left - 1; i++) {
            preNd = preNd.next; // 1 2 3 4 5
        }

        ListNode rightNd = preNd; 
        for (int i = 0; i < right - left + 1; i++) {
            rightNd = rightNd.next; // 4 5
        }

        ListNode leftNd = preNd.next; // 2 3 4 5 // rightNd.next = null = 2 3 4
        ListNode tail = rightNd.next; // 5

        preNd.next = null; // pre = 1, 
        rightNd.next = null; 

        reverse(leftNd);

        preNd.next = rightNd; //
        leftNd.next = tail;

        return dummy.next;
    }

    void reverse(ListNode head) {
        ListNode newHead = null;
        ListNode cur = head;
        while(cur!= null) {
            ListNode temp = cur.next;
            cur.next = newHead;
            newHead = cur;
            cur = temp;
        }
    }
}
```





146 LRU cache ❌

```java
public class LRUCache {
    class DLinkedNode {
        int key;
        int value;
        DLinkedNode prev;
        DLinkedNode next;
        public DLinkedNode() {}
        public DLinkedNode(int _key, int _value) {key = _key; value = _value;}
    }

    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    private int size;
    private int capacity;
    private DLinkedNode head, tail;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        moveToHead(node);
        return node.value;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (node == null) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode newNode = new DLinkedNode(key, value);
            // 添加进哈希表
            cache.put(key, newNode);
            // 添加至双向链表的头部
            addToHead(newNode);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode tail = removeTail();
                // 删除哈希表中对应的项
                cache.remove(tail.key);
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            node.value = value;
            moveToHead(node);
        }
    }

    private void addToHead(DLinkedNode node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    private void removeNode(DLinkedNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void moveToHead(DLinkedNode node) {
        removeNode(node);
        addToHead(node);
    }

    private DLinkedNode removeTail() {
        DLinkedNode res = tail.prev;
        removeNode(res);
        return res;
    }
}
```





19 remove Nth node from end of list

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(-1, head);
        ListNode temp = dummy;
        int size = 0;
        while (temp.next != null) {
            temp = temp.next;
            size++;
        }
        int pos = size - n;
        temp = dummy;
        while (pos > 0) {
            temp = temp.next;
            pos--;
        }
        temp.next = temp.next.next;
        return dummy.next;
    }
}

// 得出length
// 得出修改position
// 到当前需要修改的pos前面一位
// 然后next = next.next
// time: O(N)
// space: O(1)
```

```java
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(-1, head);
        ListNode fast = head, slow = dummy;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
}
// 双指针
// 这种题目必须要dummy，防止第一个就是需要修改的node，dummy.next可以直接修改，但是head不能直接修改头指针
// 1. 先让fast 跑到n
// 2. fast, slow 开始跑，直到fast = null, stop
// 3. slow.next = 等于n的位置，修改
// good
```



```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy = new ListNode(-1, head);
        ListNode pre = dummy;

        // find left side - 1
        for (int i = 0; i < left - 1; i++) {
            pre = pre.next;
        }
        
        // find right side - 1
        ListNode rightSide = pre;
        for (int i = 0; i < right - left + 1; i++) {
            rightSide = rightSide.next;
        }
        
        ListNode leftSide = pre.next;
        ListNode rightSideTemp = rightSide.next;

        pre.next = null;
        rightSide.next = null;
        
        reverse(leftSide); 

        // 反转之后， rightSide, leftSide 位置调换
        // rightSide 此时是头指针，leftSide 是尾指针
        pre.next = rightSide;
        leftSide.next = rightSideTemp;
        
        return dummy.next;
    }

    void reverse(ListNode head) {
        ListNode newHead = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode temp = cur.next;
            cur.next = newHead;
            newHead = cur;
            cur = temp;
        }
    }
}

// 头尾列表reverse之后， 指针的顺序也会调换
// O(N)
// O(1)
```



61 rotate list

medium

```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || head.next == null || k == 0) return head;

        int n = 1;
        ListNode temp = head;
        while (temp.next != null) {
            temp = temp.next;
            n++;
        }

        k = n - k % n;
        temp.next = head;
        while (k > 0) {
            k--;
            temp = temp.next;
        }
        ListNode res = temp.next;
        temp.next = null;
        return res;
    }
}
// 先形成环， 然后根据次序断开
// O(N)
// O(1)
```



24 swap nodes in pairs

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode dummy = new ListNode(-1 , head);
        ListNode temp = dummy;
        while (temp.next != null && temp.next.next != null) {
            ListNode first = temp.next;
            ListNode second = temp.next.next;
            temp.next = second;  
            first.next = second.next;
            second.next = first;
            temp = first;
        }
        return dummy.next;
    }
}
// 看准顺序调换就行
// O(N)
// O(1)
```



328 odd even linked list

```java
class Solution {
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode evenHead = head.next;
        ListNode odd = head, even = evenHead; 
        while (even != null && even.next != null) {
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }
}
// odd -> change odd side
// even -> change even side
// evenHead -> hold even side
// odd.next = evenHead -> link two sides
// return 

// O(N)
// O(1)
```



## Binary Search

240

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int i = matrix.length - 1, j = 0;
        while (i >=0 && j < matrix[0].length) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                i--;
            } else {
                j++;
            }
        }
        return false;
    }
}

// 从下往上 利用特性
// 如果当前第一个 > target, row--
// 如果 <= target, col++
// row++ -> 变大, col++ -> 变大

// O(logN)
// O(1)
```





74

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int low = 0, high = m * n - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            int midValue = matrix[mid / n][mid % n];
            if (target == midValue) {
                return true;
            } else if (target < midValue) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return false;
    }
}

// left <= right 很重要

// O(n)
// O(1)
```

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        
        int i = 0, j = matrix[0].length - 1;
        
        while (i < matrix.length && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }
}
// 利用特质
```





81

```java
class Solution {
    public boolean search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return true;
            
            // left sorted
            if (nums[mid] > nums[left] || nums[mid] > nums[right]) {
                if (target < nums[mid] && target >= nums[left]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
                // right sorted
            } else if (nums[mid] < nums[right] || nums[mid] < nums[left]) {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                right--;
            }
            System.out.println("left -> " + left + " right-> " + right);
        }
        return false;
    }
}
// why else { right-- } -> nums[left] = mid = right -> remove duplicate 
```





33

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            
            if (nums[mid] <= nums[right]) {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            
            if (nums[mid] >= nums[left]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return - 1;
    }
}

// while (left <= right) 很重要: [1,0] target = 0, left = 0, right = 1, target = 0, left = 1, right = 1, mid = 1 -> return mid, 如果没有这一步就直接跳过了

// O(logN)
// O(1)

```





162

```java
class Solution {
    public int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < nums[mid + 1]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        if (nums[l] > nums[r]) {
            return l;
        } else {
            return r;
        }
    }
}

// 主要是mid判断条件，如果mid > mid + 1, right = mid, else, l = mid + 1
// 此时还要在进行循环，知道left = right or left > right

// O(n)
// O(1)

```







33 search in rotated sorted array

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length -1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } 

            if (nums[right] > nums[mid]) {
                if (target <= nums[right] && target > nums[mid]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return - 1;
    }
}

// 1. left <= right 
	// 因为[0, 1] target = 1, left = 0, right = 1, mid = 0, left = mid + 1 = 1
	// left = right = 1, iteration stop
// 2. upward
	// nums[right] > nums[mid]
	// target <= nums[right], target < nums[mid]
	// 更新
// 3. downward
	// nums[right] <= nums[mid];
	// target >= nums[left], target < nums[mid]
	// 
																									
```





378 find smallest element in a sorted matrix

```java
// pq
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return a[0] - b[0]; // 这是什么意思
            }
        });

        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            pq.offer(new int[]{matrix[i][0],i, 0});
        }

        for (int i = 0; i < k - 1; i++) {
            int[] cur = pq.poll();
            if (cur[2] != n - 1) {
                pq.offer(new int[]{matrix[cur[1]][cur[2] + 1], cur[1], cur[2] + 1});
            }
        }
        return pq.poll()[0]; 
    }
}
```



```java
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        int m = matrix.length;
        int n = matrix[0].length;
        int index = 0;
        int[] array = new int[m * n];
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < n; b++) {
                array[index] = matrix[a][b];
                index++;
            }
        }  
        Arrays.sort(array);
        return array[k-1];
    }
}

// O(n2logN)
// O(n2) need n2 to save the array
```



## Heap

```java
public int[] smallestRange(List<List<Integer>> a) {
        PriorityQueue<int[]> q = new PriorityQueue<>(Comparator.comparingInt(o -> a.get(o[0]).get(o[1])));
  // max 保存的就是当前第一个数字的最大数字
  // start 就是一个start
  // end 就是不断的min取最小数字
        int max = Integer.MIN_VALUE, start = 0, end = Integer.MAX_VALUE;
  
  // 先添加array q:[[0,0], [1,0], [2,0]]
  // max = 5
        for (int i = 0; i < a.size(); i++) {
            q.offer(new int[]{i, 0});
            max = Math.max(max, a.get(i).get(0));
        }
// q.offer(0, 1) [1, 0], [2, 0]
        while (q.size() == a.size()) {
			
          // [0,1]
            int e[] = q.poll(), row = e[0], col = e[1];
    				

 
          // start = 10
          // end = 10;
            if (end - start > max - a.get(row).get(col)) {
                start = a.get(row).get(col);
                end = max;
            }
          

            if (col + 1 < a.get(row).size()) {
                q.offer(new int[]{row, col + 1});
                max = Math.max(max, a.get(row).get(col + 1));
            }
        }
        return new int[]{start, end};
    }

// 一开始保存的东西都是各各array的具体位置，
// 然后通过一种更新的方法，来保存一个range里面的信息，然后通过遍历 不断更新位置的信息，
// 更新位置信息之后，end 是上一个位置的最大值， 然后start是当前的最大值
// 
```





767 reorganize strings

```java
class Solution {
    public String reorganizeString(String s) {
        Map<Character, Integer> map = new HashMap<>();
        
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        
        PriorityQueue<Character> pq = new PriorityQueue<Character>((a,b) -> map.get(b) - map.get(a));
        
        pq.addAll(map.keySet());
        
        StringBuilder sb = new StringBuilder();
        
        while(pq.size() > 1) {
            char first = pq.poll();
            char second = pq.poll();
            sb.append(first);
            sb.append(second);
            
            map.put(first, map.get(first) - 1);
            map.put(second, map.get(second) - 1);
            
            if (map.get(first) > 0) {
                pq.offer(first);
            }
            if (map.get(second) > 0) {
                pq.offer(second);
            }
        }
        
        while (!pq.isEmpty()) {
            if (map.get(pq.peek()) > 1) {
                return "";
            } else {
                sb.append(pq.poll());
            }
        }
        return sb.toString();
    }
}

// 1. 按照出现次数排序，插入进PQ中，（pq插入的东西是char）
// 2. 取出frequency 出现次数排名1，2的char
// 3. sb.append()
// 4. 如果剩下的一个!pq.isEmpty() & pq.peek() frequency > 2 -> "空字符串"， sb.append()
```





451 sort characters by frequency

```java
class Solution {
    public String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>((a, b) -> b.getValue() - a.getValue());
        
        pq.addAll(map.entrySet());
        
        StringBuilder sb = new StringBuilder();
        
        while (!pq.isEmpty()) {
            Map.Entry a = pq.poll();
            int freq = (int) a.getValue();
            while (freq-- > 0) {
                sb.append(a.getKey());
            }
        }
        return sb.toString();
    }
}
// 怎么创建pq， 怎么往pq里面添加元素是最重要的

// O(n)
// O(n)
```





347 top k frequent element

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i : nums) {
            map.put(i, map.getOrDefault(i, 0) + 1);
        }
        
        PriorityQueue<Map.Entry<Integer, Integer>> pq = new PriorityQueue<>((a, b) -> b.getValue() - a.getValue());
        
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            pq.offer(entry);
        }
        
        int i = 0;
        int[] res = new int[k];
        while (i < k) {
            Map.Entry<Integer, Integer> a = pq.poll();
            res[i++] = a.getKey();
        }
        return res;
        
    }
}

// Map.Entry<Integer, Integer> 就是每一层的入口
// 经典套路就是怎么把key value 结合塞进map里面

// O(n)
// O(k)
```





23 merge k sorted list

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) return null;
        ListNode head = new ListNode(-1);
        ListNode tail = head;

        PriorityQueue<ListNode> pq = new PriorityQueue<ListNode>(lists.length, 
        new Comparator<ListNode>() {
            public int compare(ListNode v1, ListNode v2) {
                return v1.val - v2.val;
            }
        });


        for (ListNode list : lists) {
            if (list != null) pq.offer(list);
        }

        while (!pq.isEmpty()) {
            ListNode cur = pq.poll();
            tail.next = cur;
            tail = tail.next;

            if (cur.next != null) {
                pq.offer(cur.next);
            }
        }
        return head.next;
    }
}

// 1. create the correct pq
// 2. add ListNode list to pq
// 3. put list.next to pq, and add pq.val to head

// O(kn logK)
// O(K)
```





215 kth largest element in an array

Medium

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
    int len=nums.length;
    PriorityQueue<Integer> queue=new PriorityQueue<>((a, b) -> a - b);
    for(int i=0;i<len;i++){
        if(queue.size()<k){
            queue.offer(nums[i]);
        }else{
            if(queue.peek()>=nums[i]) continue;
            else{
                queue.poll();
                queue.offer(nums[i]);
            }
        }
    }
    return queue.peek();
    }
}

// 1. create一个最小堆, 堆的大小是k
// 2. 如果pq.peek() > nums[i], 推入当前元素
// 3. 因为是最小堆，所以顶端元素一定是kth里面最小的那个元素

// O(N)
// O(k)
```







378 k th smallest element in a sorted matrix

Medium

```java
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((o1, o2) -> (o1[0] - o2[0]));
     	// PriorityQueue<int[]> pq = new PriorityQueue<int[]>(new Comparator<int[]>() {
      //      public int compare(int[] a, int[] b) {
      //          return a[0] - b[0];
      //      }
      //  });

        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            pq.offer(new int[]{matrix[i][0],i, 0});
        }

        for (int i = 0; i < k - 1; i++) {
            int[] cur = pq.poll();
            if (cur[2] != n - 1) {
                pq.offer(new int[]{matrix[cur[1]][cur[2] + 1], cur[1], cur[2] + 1});
            }
        }
        return pq.poll()[0]; 
    }
}

// Java 语法， A, 1, B, 2 小的在上面，ascending order 排序

// 1. 创建Q， Q的排列顺序是按照list的一个数字的大小分类的，小的在上面
// 2. offer所有row在Q里，然后添加入新的col+1在Q里，会自动分序
// 3. 遍历到K-1
// 4. poll出，得到matrix的value
```



373 find k pairs with smallest sums

```java
class Solution {
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> pairs = new ArrayList<>();

        if (nums1.length == 0 || nums2.length == 0 || k == 0) return pairs;

        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0] + a[1] - b[1]);
        
        for (int i = 0; i < nums1.length && i < k; i++) {
            pq.offer(new int[]{nums1[i], nums2[0], 0});
        }

        while (k-- > 0 && !pq.isEmpty()) {
            int[] cur = pq.poll();
            List<Integer> pair = new ArrayList<>();
            pair.add(cur[0]);
            pair.add(cur[1]);
            pairs.add(pair);
            if (cur[2] < nums2.length - 1) {
                pq.offer(new int[]{cur[0], nums2[cur[2] + 1], cur[2] + 1});
            }
        }
        return pairs;
    }
}

// 1. create PriorityQueue
// 2. put nums1[0], nums1[1]...nums1[n] with num2[0] into the pq (seed some random data)
// 3. k-- ? , nums1[0], nums2[0], must in kth smallest pair
// 		so, while put nums1[0], nums2[0] in pair, k = k -1, also put nums1[0] nums2[1] into 
//		the pq, and next put nums1[1],nums2[1] into the deck...
// 		until get k pairs that satisfy the requirement.

// O(klog(k))
// O(size of the priority queue)
```







## String

415 add string 

Easy

```java
class Solution {
    public String addStrings(String num1, String num2) {
        StringBuilder sb = new StringBuilder();
        int n1 = num1.length() - 1, n2 = num2.length() - 1, carry = 0;
        while (n1 >= 0 || n2 >= 0 || carry != 0) {
            int a = n1 >= 0 ? num1.charAt(n1) - '0' : 0;
            int b = n2 >= 0 ? num2.charAt(n2) - '0' : 0;
            int sum = a + b + carry;
            sb.append(sum % 10);
            carry = sum / 10;
            n1--;
            n2--;
        }
        return sb.reverse().toString();
    }
}
// 注意n1 >= 0, n2 >= 0, carry != 0
// 所有的条件都要link 0，很重要

// O(max(M + N))
// O(1)
```



## Greedy

435 non-overlapping intervals

```java
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        int n = intervals.length;
        if (intervals == null || n == 0) return 0;
        
        Arrays.sort(intervals, (a,b) -> a[1] - b[1]);
        
        int preEnd = intervals[0][1];
        int count = 0;
        
        for (int i = 1; i < n; i++) {
            // end > preEnd, start < preEnd, ++
            if (intervals[i][0] < preEnd) {
                count++;
            } else {
                // start >= preEnd, not add, revise preEnd;
                preEnd = intervals[i][1];
            }
        }
        return count;
    }
}

// O(nlogn)
// O(1)
```



56 merge intervals

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        int n = intervals.length;
        List<int[]> res = new ArrayList<>();
        
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        
        int[] pre = intervals[0];
        
        for (int i = 1; i < n; i++) {
            // start > preEnd 
            if (intervals[i][0] > pre[1]) {
                res.add(pre);
                pre = intervals[i];
                // start <= preEnd
            } else {
                pre[1] = Math.max(pre[1], intervals[i][1]);
            }
        }
        res.add(pre);
        return res.toArray(new int[0][]);
    }
}

// 1. sort by start
// 2. if start > preEnd, add pre
// 3. if start <= preEnd, 修改pre
// 4. 最后别忘记添加pre

// O(nlogn)
// O(k)
```



## Trie

720

```java
class Solution {
    class Node {
        public String s;
        public boolean isWord;
        public Node[] children;
        
        public Node() {
            this.children = new Node[26];
        }
    }
    
    class Trie {
        private Node root;
        
        public Trie() {
            this.root = new Node();
        }
        
        public void insert(String word) {
            Node cur = root;
            for (int i = 0; i < word.length(); i++) {
                int idx = word.charAt(i) - 'a';
                if (cur.children[idx] == null) {
                    cur.children[idx] = new Node();
                } 
                cur = cur.children[idx];
            }
            cur.isWord = true;
            cur.s = word;
        }
        
        public String findLongest() {
            String res = "";
            Queue<Node> q = new LinkedList<>();
            q.offer(root);
            while(!q.isEmpty()) {
                int size = q.size();
                for (int i = 0; i < size; i++) {
                    Node nd = q.poll();
                    
                  // start from end ? 
                    for (int j = 25; j >= 0; j--) {
                        if (nd.children[j] != null && nd.children[j].isWord) {
                            res = nd.children[j].s;
                            q.offer(nd.children[j]);
                        }
                    }
                }
            }
            return res;
        }
    }
    
    public String longestWord(String[] words) {
        Trie t = new Trie();
        
        for (String word : words) {
            t.insert(word);
        }
        
        return t.findLongest();
    }
}
```





208

```java
class Trie {
    
    private Node root;

    public Trie() {
        root = new Node('\0');
    }
    
    public void insert(String word) {
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (cur.children[c - 'a'] == null) {
                cur.children[c - 'a'] = new Node(c);    
            } 
            cur = cur.children[c - 'a'];
        }
        cur.isWord = true;
    }
    
    public boolean search(String word) {
        Node nd = getNode(word);
       return  nd != null && nd.isWord;
    }
    
    public boolean startsWith(String prefix) {
        Node nd = getNode(prefix);
        return nd != null;
    }
    
    private Node getNode(String word) {
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (cur.children[c - 'a'] == null) {
                return null;
            } else {
                cur = cur.children[c - 'a'];
            }
        }
        return cur;
    }
    
    class Node {
        public char c;
        public boolean isWord;
        public Node[] children;
        
        public Node(char c) {
            this.c = c;
            this.isWord = false;
            this.children = new Node[26];
        }
    }
}

// 主要是一个data structure 的运用
```







# Note

## Recursion

```java
n = {root = target, map.put(0)}
  = {root.left = target, map.put(root,1), map.put(root.left, 0)}
	= {root.right = target, map.put(root,1), map.put(root.right, 0)}

  5(2)
 4(1)   7 -> buy using length
3 5(0) 6 8

void find(TreeNode root, TreeNode target) {
  
}
```







## Bianbian



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



## YM

315 Count of Smaller Numbers After Self / Algorithm swap
493 reverse pairs
829 consecutive numbers sum
994 rotting orange

1043. Partition Array for Maximum Sum
1335 Minimum Difficulty of a Job Schedule



21: Merge two sorted Lists ✅
200: Number of Islands ✅
547: number of province
692: top K frequent word
937: reorder data in log files
973: K closest point
1010: Pairs of Songs With Total Durations Divisible by 60 / Amazon Music
1041: robot bounded in circle
1099: Two Sum Less Than K
1135: Connecting Cities With Minimum Cost (MST)
1167: Minimum Cost to Connect Sticks
1197: Minimum Knight Moves / Demolition Robot
1465: Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts / Storage Optimize
1629: Slowest key
1710:Maximum Units on a Truck

