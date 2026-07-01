'''
Date: 2026-01-26 14:35:39
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-01-26 15:22:20
FilePath: \leetcode\搜索插入位置.py
'''
def searchInsert(nums, target) -> int:
    l = 0
    r = len(nums)-1
    while l<=r:
        mid = int((r+l)/2)
        if nums[mid]>target:
            r=mid-1
        else:
            l=mid+1
    return r
nums = [1,2,3,4,5,6,8]
print(searchInsert(nums,7))