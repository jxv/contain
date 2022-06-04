module Data.Contain where

import Prelude

import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.IntMap as IM
import qualified Data.IntSet as IntSet

{-
import Data.Char (toUpper, toLower)

rename :: String -> FilePath -> FilePath -> IO ()
rename prefix src des = do
  c <- readFile src
  writeFile des $ xform (length prefix) c
  where
    xform :: Int -> String -> String
    xform i s = let
      newDefine l = let
        spanned :: (String, String)
        spanned = span (/= ' ') l
        --
        qualifiedName :: String
        qualifiedName = map toLower $ take i (fst spanned)
        --
        originalName :: String
        originalName = drop (i + 1) (fst spanned)
        --
        upperFirstChar :: String
        upperFirstChar = toUpper (head originalName): tail originalName
        --
        newName :: String
        newName = qualifiedName ++ upperFirstChar
        --
        in [newName ++ snd spanned, newName ++ " = " ++ fst spanned]
      in unlines $ map unlines (map newDefine (lines s))
-}

--
--
--

type Set = S.Set

sAlterF :: (Ord a, Functor f) => (Bool -> f Bool) -> a -> Set a -> f (Set a)
sAlterF = S.alterF

sCartesianProduct :: Set a -> Set b -> Set (a, b)
sCartesianProduct = S.cartesianProduct

sDelete :: Ord a => a -> Set a -> Set a
sDelete = S.delete

sDeleteAt :: Int -> Set a -> Set a
sDeleteAt = S.deleteAt

sDeleteFindMax :: Set a -> (a, Set a)
sDeleteFindMax = S.deleteFindMax

sDeleteFindMin :: Set a -> (a, Set a)
sDeleteFindMin = S.deleteFindMin

sDeleteMax :: Set a -> Set a
sDeleteMax = S.deleteMax

sDeleteMin :: Set a -> Set a
sDeleteMin = S.deleteMin

sDifference :: Ord a => Set a -> Set a -> Set a
sDifference = S.difference

sDisjoint :: Ord a => Set a -> Set a -> Bool
sDisjoint = S.disjoint

sDisjointUnion :: Set a -> Set b -> Set (Either a b)
sDisjointUnion = S.disjointUnion

sDrop :: Int -> Set a -> Set a
sDrop = S.drop

sDropWhileAntitone :: (a -> Bool) -> Set a -> Set a
sDropWhileAntitone = S.dropWhileAntitone

sElemAt :: Int -> Set a -> a
sElemAt = S.elemAt

sElems :: Set a -> [a]
sElems = S.elems

sEmpty :: Set a
sEmpty = S.empty

sFilter :: (a -> Bool) -> Set a -> Set a
sFilter = S.filter

sFindIndex :: Ord a => a -> Set a -> Int
sFindIndex = S.findIndex

sFindMax :: Set a -> a
sFindMax = S.findMax

sFindMin :: Set a -> a
sFindMin = S.findMin

sFold :: (a -> b -> b) -> b -> Set a -> b
sFold = S.fold

sFoldl :: (a -> b -> a) -> a -> Set b -> a
sFoldl = S.foldl

sFoldl' :: (a -> b -> a) -> a -> Set b -> a
sFoldl' = S.foldl'

sFoldr :: (a -> b -> b) -> b -> Set a -> b
sFoldr = S.foldr

sFoldr' :: (a -> b -> b) -> b -> Set a -> b
sFoldr' = S.foldr'

sFromAscList :: Eq a => [a] -> Set a
sFromAscList = S.fromAscList

sFromDescList :: Eq a => [a] -> Set a
sFromDescList = S.fromDescList

sFromDistinctAscList :: [a] -> Set a
sFromDistinctAscList = S.fromDistinctAscList

sFromDistinctDescList :: [a] -> Set a
sFromDistinctDescList = S.fromDistinctDescList

sFromList :: Ord a => [a] -> Set a
sFromList = S.fromList

sInsert :: Ord a => a -> Set a -> Set a
sInsert = S.insert

sIntersection :: Ord a => Set a -> Set a -> Set a
sIntersection = S.intersection

sIsProperSubsetOf :: Ord a => Set a -> Set a -> Bool
sIsProperSubsetOf = S.isProperSubsetOf

sIsSubsetOf :: Ord a => Set a -> Set a -> Bool
sIsSubsetOf = S.isSubsetOf

sLookupGE :: Ord a => a -> Set a -> Maybe a
sLookupGE = S.lookupGE

sLookupGT :: Ord a => a -> Set a -> Maybe a
sLookupGT = S.lookupGT

sLookupIndex :: Ord a => a -> Set a -> Maybe Int
sLookupIndex = S.lookupIndex

sLookupLE :: Ord a => a -> Set a -> Maybe a
sLookupLE = S.lookupLE

sLookupLT :: Ord a => a -> Set a -> Maybe a
sLookupLT = S.lookupLT

sLookupMax :: Set a -> Maybe a
sLookupMax = S.lookupMax

sLookupMin :: Set a -> Maybe a
sLookupMin = S.lookupMin

sMap :: Ord b => (a -> b) -> Set a -> Set b
sMap = S.map

sMapMonotonic :: (a -> b) -> Set a -> Set b
sMapMonotonic = S.mapMonotonic

sMaxView :: Set a -> Maybe (a, Set a)
sMaxView = S.maxView

sMember :: Ord a => a -> Set a -> Bool
sMember = S.member

sMinView :: Set a -> Maybe (a, Set a)
sMinView = S.minView

sNotMember :: Ord a => a -> Set a -> Bool
sNotMember = S.notMember

sNull :: Set a -> Bool
sNull = S.null

sPartition :: (a -> Bool) -> Set a -> (Set a, Set a)
sPartition = S.partition

sPowerSet :: Set a -> Set (Set a)
sPowerSet = S.powerSet

sShowTree :: Show a => Set a -> String
sShowTree = S.showTree

sShowTreeWith :: Show a => Bool -> Bool -> Set a -> String
sShowTreeWith = S.showTreeWith

sSingleton :: a -> Set a
sSingleton = S.singleton

sSize :: Set a -> Int
sSize = S.size

sSpanAntitone :: (a -> Bool) -> Set a -> (Set a, Set a)
sSpanAntitone = S.spanAntitone

sSplit :: Ord a => a -> Set a -> (Set a, Set a)
sSplit = S.split

sSplitAt :: Int -> Set a -> (Set a, Set a)
sSplitAt = S.splitAt

sSplitMember :: Ord a => a -> Set a -> (Set a, Bool, Set a)
sSplitMember = S.splitMember

sSplitRoot :: Set a -> [Set a]
sSplitRoot = S.splitRoot

sTake :: Int -> Set a -> Set a
sTake = S.take

sTakeWhileAntitone :: (a -> Bool) -> Set a -> Set a
sTakeWhileAntitone = S.takeWhileAntitone

sToAscList :: Set a -> [a]
sToAscList = S.toAscList

sToDescList :: Set a -> [a]
sToDescList = S.toDescList

sToList :: Set a -> [a]
sToList = S.toList

sUnion :: Ord a => Set a -> Set a -> Set a
sUnion = S.union

sUnions :: (Foldable f, Ord a) => f (Set a) -> Set a
sUnions = S.unions

sValid :: Ord a => Set a -> Bool
sValid = S.valid

--
--
--

type Map = M.Map

mAdjust :: Ord k => (a -> a) -> k -> Map k a -> Map k a
mAdjust = M.adjust

mAdjustWithKey :: Ord k => (k -> a -> a) -> k -> Map k a -> Map k a
mAdjustWithKey = M.adjustWithKey

mAlter :: Ord k => (Maybe a -> Maybe a) -> k -> Map k a -> Map k a
mAlter = M.alter

mAlterF :: (Functor f, Ord k) => (Maybe a -> f (Maybe a)) -> k -> Map k a -> f (Map k a)
mAlterF = M.alterF

mAssocs :: Map k a -> [(k, a)]
mAssocs = M.assocs

mCompose :: Ord b => Map b c -> Map a b -> Map a c
mCompose = M.compose

mDelete :: Ord k => k -> Map k a -> Map k a
mDelete = M.delete

mDeleteAt :: Int -> Map k a -> Map k a
mDeleteAt = M.deleteAt

mDeleteFindMax :: Map k a -> ((k, a), Map k a)
mDeleteFindMax = M.deleteFindMax

mDeleteFindMin :: Map k a -> ((k, a), Map k a)
mDeleteFindMin = M.deleteFindMin

mDeleteMax :: Map k a -> Map k a
mDeleteMax = M.deleteMax

mDeleteMin :: Map k a -> Map k a
mDeleteMin = M.deleteMin

mDifference :: Ord k => Map k a -> Map k b -> Map k a
mDifference = M.difference

mDifferenceWith :: Ord k => (a -> b -> Maybe a) -> Map k a -> Map k b -> Map k a
mDifferenceWith = M.differenceWith

mDifferenceWithKey :: Ord k => (k -> a -> b -> Maybe a) -> Map k a -> Map k b -> Map k a
mDifferenceWithKey = M.differenceWithKey

mDisjoint :: Ord k => Map k a -> Map k b -> Bool
mDisjoint = M.disjoint

mDrop :: Int -> Map k a -> Map k a
mDrop = M.drop

mDropWhileAntitone :: (k -> Bool) -> Map k a -> Map k a
mDropWhileAntitone = M.dropWhileAntitone

mElemAt :: Int -> Map k a -> (k, a)
mElemAt = M.elemAt

mElems :: Map k a -> [a]
mElems = M.elems

mEmpty :: Map k a
mEmpty = M.empty

mFilter :: (a -> Bool) -> Map k a -> Map k a
mFilter = M.filter

mFilterWithKey :: (k -> a -> Bool) -> Map k a -> Map k a
mFilterWithKey = M.filterWithKey

mFindIndex :: Ord k => k -> Map k a -> Int
mFindIndex = M.findIndex

mFindMax :: Map k a -> (k, a)
mFindMax = M.findMax

mFindMin :: Map k a -> (k, a)
mFindMin = M.findMin

mFindWithDefault :: Ord k => a -> k -> Map k a -> a
mFindWithDefault = M.findWithDefault

mFoldMapWithKey :: Monoid m => (k -> a -> m) -> Map k a -> m
mFoldMapWithKey = M.foldMapWithKey

mFoldl :: (a -> b -> a) -> a -> Map k b -> a
mFoldl = M.foldl

mFoldl' :: (a -> b -> a) -> a -> Map k b -> a
mFoldl' = M.foldl'

mFoldlWithKey :: (a -> k -> b -> a) -> a -> Map k b -> a
mFoldlWithKey = M.foldlWithKey

mFoldlWithKey' :: (a -> k -> b -> a) -> a -> Map k b -> a
mFoldlWithKey' = M.foldlWithKey'

mFoldr :: (a -> b -> b) -> b -> Map k a -> b
mFoldr = M.foldr

mFoldr' :: (a -> b -> b) -> b -> Map k a -> b
mFoldr' = M.foldr'

mFoldrWithKey :: (k -> a -> b -> b) -> b -> Map k a -> b
mFoldrWithKey = M.foldrWithKey

mFoldrWithKey' :: (k -> a -> b -> b) -> b -> Map k a -> b
mFoldrWithKey' = M.foldrWithKey'

mFromAscList :: Eq k => [(k, a)] -> Map k a
mFromAscList = M.fromAscList

mFromAscListWith :: Eq k => (a -> a -> a) -> [(k, a)] -> Map k a
mFromAscListWith = M.fromAscListWith

mFromAscListWithKey :: Eq k => (k -> a -> a -> a) -> [(k, a)] -> Map k a
mFromAscListWithKey = M.fromAscListWithKey

mFromDescList :: Eq k => [(k, a)] -> Map k a
mFromDescList = M.fromDescList

mFromDescListWith :: Eq k => (a -> a -> a) -> [(k, a)] -> Map k a
mFromDescListWith = M.fromDescListWith

mFromDescListWithKey :: Eq k => (k -> a -> a -> a) -> [(k, a)] -> Map k a
mFromDescListWithKey = M.fromDescListWithKey

mFromDistinctAscList :: [(k, a)] -> Map k a
mFromDistinctAscList = M.fromDistinctAscList

mFromDistinctDescList :: [(k, a)] -> Map k a
mFromDistinctDescList = M.fromDistinctDescList

mFromList :: Ord k => [(k, a)] -> Map k a
mFromList = M.fromList

mFromListWith :: Ord k => (a -> a -> a) -> [(k, a)] -> Map k a
mFromListWith = M.fromListWith

mFromListWithKey :: Ord k => (k -> a -> a -> a) -> [(k, a)] -> Map k a
mFromListWithKey = M.fromListWithKey

mFromSet :: (k -> a) -> Set k -> Map k a
mFromSet = M.fromSet

mInsert :: Ord k => k -> a -> Map k a -> Map k a
mInsert = M.insert

mInsertLookupWithKey :: Ord k => (k -> a -> a -> a) -> k -> a -> Map k a -> (Maybe a, Map k a)
mInsertLookupWithKey = M.insertLookupWithKey

mInsertWith :: Ord k => (a -> a -> a) -> k -> a -> Map k a -> Map k a
mInsertWith = M.insertWith

mInsertWithKey :: Ord k => (k -> a -> a -> a) -> k -> a -> Map k a -> Map k a
mInsertWithKey = M.insertWithKey

mIntersection :: Ord k => Map k a -> Map k b -> Map k a
mIntersection = M.intersection

mIntersectionWith :: Ord k => (a -> b -> c) -> Map k a -> Map k b -> Map k c
mIntersectionWith = M.intersectionWith

mIntersectionWithKey :: Ord k => (k -> a -> b -> c) -> Map k a -> Map k b -> Map k c
mIntersectionWithKey = M.intersectionWithKey

mIsProperSubmapOf :: (Ord k, Eq a) => Map k a -> Map k a -> Bool
mIsProperSubmapOf = M.isProperSubmapOf

mIsProperSubmapOfBy :: Ord k => (a -> b -> Bool) -> Map k a -> Map k b -> Bool
mIsProperSubmapOfBy = M.isProperSubmapOfBy

mIsSubmapOf :: (Ord k, Eq a) => Map k a -> Map k a -> Bool
mIsSubmapOf = M.isSubmapOf

mIsSubmapOfBy :: Ord k => (a -> b -> Bool) -> Map k a -> Map k b -> Bool
mIsSubmapOfBy = M.isSubmapOfBy

mKeys :: Map k a -> [k]
mKeys = M.keys

mKeysSet :: Map k a -> Set k
mKeysSet = M.keysSet

mLookup :: Ord k => k -> Map k a -> Maybe a
mLookup = M.lookup

mLookupGE :: Ord k => k -> Map k v -> Maybe (k, v)
mLookupGE = M.lookupGE

mLookupGT :: Ord k => k -> Map k v -> Maybe (k, v)
mLookupGT = M.lookupGT

mLookupIndex :: Ord k => k -> Map k a -> Maybe Int
mLookupIndex = M.lookupIndex

mLookupLE :: Ord k => k -> Map k v -> Maybe (k, v)
mLookupLE = M.lookupLE

mLookupLT :: Ord k => k -> Map k v -> Maybe (k, v)
mLookupLT = M.lookupLT

mLookupMax :: Map k a -> Maybe (k, a)
mLookupMax = M.lookupMax

mLookupMin :: Map k a -> Maybe (k, a)
mLookupMin = M.lookupMin

mMap :: (a -> b) -> Map k a -> Map k b
mMap = M.map

mMapAccum :: (a -> b -> (a, c)) -> a -> Map k b -> (a, Map k c)
mMapAccum = M.mapAccum

mMapAccumRWithKey :: (a -> k -> b -> (a, c)) -> a -> Map k b -> (a, Map k c)
mMapAccumRWithKey = M.mapAccumRWithKey

mMapAccumWithKey :: (a -> k -> b -> (a, c)) -> a -> Map k b -> (a, Map k c)
mMapAccumWithKey = M.mapAccumWithKey

mMapEither :: (a -> Either b c) -> Map k a -> (Map k b, Map k c)
mMapEither = M.mapEither

mMapEitherWithKey :: (k -> a -> Either b c) -> Map k a -> (Map k b, Map k c)
mMapEitherWithKey = M.mapEitherWithKey

mMapKeys :: Ord k2 => (k1 -> k2) -> Map k1 a -> Map k2 a
mMapKeys = M.mapKeys

mMapKeysMonotonic :: (k1 -> k2) -> Map k1 a -> Map k2 a
mMapKeysMonotonic = M.mapKeysMonotonic

mMapKeysWith :: Ord k2 => (a -> a -> a) -> (k1 -> k2) -> Map k1 a -> Map k2 a
mMapKeysWith = M.mapKeysWith

mMapMaybe :: (a -> Maybe b) -> Map k a -> Map k b
mMapMaybe = M.mapMaybe

mMapMaybeWithKey :: (k -> a -> Maybe b) -> Map k a -> Map k b
mMapMaybeWithKey = M.mapMaybeWithKey

mMapWithKey :: (k -> a -> b) -> Map k a -> Map k b
mMapWithKey = M.mapWithKey

mMaxView :: Map k a -> Maybe (a, Map k a)
mMaxView = M.maxView

mMaxViewWithKey :: Map k a -> Maybe ((k, a), Map k a)
mMaxViewWithKey = M.maxViewWithKey

mMember :: Ord k => k -> Map k a -> Bool
mMember = M.member

mMergeWithKey :: Ord k => (k -> a -> b -> Maybe c) -> (Map k a -> Map k c) -> (Map k b -> Map k c) -> Map k a -> Map k b -> Map k c
mMergeWithKey = M.mergeWithKey

mMinView :: Map k a -> Maybe (a, Map k a)
mMinView = M.minView

mMinViewWithKey :: Map k a -> Maybe ((k, a), Map k a)
mMinViewWithKey = M.minViewWithKey

mNotMember :: Ord k => k -> Map k a -> Bool
mNotMember = M.notMember

mNull :: Map k a -> Bool
mNull = M.null

mPartition :: (a -> Bool) -> Map k a -> (Map k a, Map k a)
mPartition = M.partition

mPartitionWithKey :: (k -> a -> Bool) -> Map k a -> (Map k a, Map k a)
mPartitionWithKey = M.partitionWithKey

mRestrictKeys :: Ord k => Map k a -> Set k -> Map k a
mRestrictKeys = M.restrictKeys

mSingleton :: k -> a -> Map k a
mSingleton = M.singleton

mSize :: Map k a -> Int
mSize = M.size

mSpanAntitone :: (k -> Bool) -> Map k a -> (Map k a, Map k a)
mSpanAntitone = M.spanAntitone

mSplit :: Ord k => k -> Map k a -> (Map k a, Map k a)
mSplit = M.split

mSplitAt :: Int -> Map k a -> (Map k a, Map k a)
mSplitAt = M.splitAt

mSplitLookup :: Ord k => k -> Map k a -> (Map k a, Maybe a, Map k a)
mSplitLookup = M.splitLookup

mSplitRoot :: Map k b -> [Map k b]
mSplitRoot = M.splitRoot

mTake :: Int -> Map k a -> Map k a
mTake = M.take

mTakeWhileAntitone :: (k -> Bool) -> Map k a -> Map k a
mTakeWhileAntitone = M.takeWhileAntitone

mToAscList :: Map k a -> [(k, a)]
mToAscList = M.toAscList

mToDescList :: Map k a -> [(k, a)]
mToDescList = M.toDescList

mToList :: Map k a -> [(k, a)]
mToList = M.toList

mTraverseMaybeWithKey :: Applicative f => (k -> a -> f (Maybe b)) -> Map k a -> f (Map k b)
mTraverseMaybeWithKey = M.traverseMaybeWithKey

mTraverseWithKey :: Applicative t => (k -> a -> t b) -> Map k a -> t (Map k b)
mTraverseWithKey = M.traverseWithKey

mUnion :: Ord k => Map k a -> Map k a -> Map k a
mUnion = M.union

mUnionWith :: Ord k => (a -> a -> a) -> Map k a -> Map k a -> Map k a
mUnionWith = M.unionWith

mUnionWithKey :: Ord k => (k -> a -> a -> a) -> Map k a -> Map k a -> Map k a
mUnionWithKey = M.unionWithKey

mUnions :: (Foldable f, Ord k) => f (Map k a) -> Map k a
mUnions = M.unions

mUnionsWith :: (Foldable f, Ord k) => (a -> a -> a) -> f (Map k a) -> Map k a
mUnionsWith = M.unionsWith

mUpdate :: Ord k => (a -> Maybe a) -> k -> Map k a -> Map k a
mUpdate = M.update

mUpdateAt :: (k -> a -> Maybe a) -> Int -> Map k a -> Map k a
mUpdateAt = M.updateAt

mUpdateLookupWithKey :: Ord k => (k -> a -> Maybe a) -> k -> Map k a -> (Maybe a, Map k a)
mUpdateLookupWithKey = M.updateLookupWithKey

mUpdateMax :: (a -> Maybe a) -> Map k a -> Map k a
mUpdateMax = M.updateMax

mUpdateMaxWithKey :: (k -> a -> Maybe a) -> Map k a -> Map k a
mUpdateMaxWithKey = M.updateMaxWithKey

mUpdateMin :: (a -> Maybe a) -> Map k a -> Map k a
mUpdateMin = M.updateMin

mUpdateMinWithKey :: (k -> a -> Maybe a) -> Map k a -> Map k a
mUpdateMinWithKey = M.updateMinWithKey

mUpdateWithKey :: Ord k => (k -> a -> Maybe a) -> k -> Map k a -> Map k a
mUpdateWithKey = M.updateWithKey

mValid :: Ord k => Map k a -> Bool
mValid = M.valid

mWithoutKeys :: Ord k => Map k a -> Set k -> Map k a
mWithoutKeys = M.withoutKeys

--
--
--

type IntSet = IntSet.IntSet

intsetAlterF :: Functor f => (Bool -> f Bool) -> Int -> IntSet -> f IntSet
intsetAlterF = IntSet.alterF

intsetDelete :: Int -> IntSet -> IntSet
intsetDelete = IntSet.delete

intsetDeleteFindMax :: IntSet -> (Int, IntSet)
intsetDeleteFindMax = IntSet.deleteFindMax

intsetDeleteFindMin :: IntSet -> (Int, IntSet)
intsetDeleteFindMin = IntSet.deleteFindMin

intsetDeleteMax :: IntSet -> IntSet
intsetDeleteMax = IntSet.deleteMax

intsetDeleteMin :: IntSet -> IntSet
intsetDeleteMin = IntSet.deleteMin

intsetDifference :: IntSet -> IntSet -> IntSet
intsetDifference = IntSet.difference

intsetDisjoint :: IntSet -> IntSet -> Bool
intsetDisjoint = IntSet.disjoint

intsetElems :: IntSet -> [Int]
intsetElems = IntSet.elems

intsetEmpty :: IntSet
intsetEmpty = IntSet.empty

intsetFilter :: (Int -> Bool) -> IntSet -> IntSet
intsetFilter = IntSet.filter

intsetFindMax :: IntSet -> Int
intsetFindMax = IntSet.findMax

intsetFindMin :: IntSet -> Int
intsetFindMin = IntSet.findMin

intsetFold :: (Int -> b -> b) -> b -> IntSet -> b
intsetFold = IntSet.fold

intsetFoldl :: (a -> Int -> a) -> a -> IntSet -> a
intsetFoldl = IntSet.foldl

intsetFoldl' :: (a -> Int -> a) -> a -> IntSet -> a
intsetFoldl' = IntSet.foldl'

intsetFoldr :: (Int -> b -> b) -> b -> IntSet -> b
intsetFoldr = IntSet.foldr

intsetFoldr' :: (Int -> b -> b) -> b -> IntSet -> b
intsetFoldr' = IntSet.foldr'

intsetFromAscList :: [Int] -> IntSet
intsetFromAscList = IntSet.fromAscList

intsetFromDistinctAscList :: [Int] -> IntSet
intsetFromDistinctAscList = IntSet.fromDistinctAscList

intsetFromList :: [Int] -> IntSet
intsetFromList = IntSet.fromList

intsetInsert :: Int -> IntSet -> IntSet
intsetInsert = IntSet.insert

intsetIntersection :: IntSet -> IntSet -> IntSet
intsetIntersection = IntSet.intersection

intsetIsProperSubsetOf :: IntSet -> IntSet -> Bool
intsetIsProperSubsetOf = IntSet.isProperSubsetOf

intsetIsSubsetOf :: IntSet -> IntSet -> Bool
intsetIsSubsetOf = IntSet.isSubsetOf

intsetLookupGE :: Int -> IntSet -> Maybe Int
intsetLookupGE = IntSet.lookupGE

intsetLookupGT :: Int -> IntSet -> Maybe Int
intsetLookupGT = IntSet.lookupGT

intsetLookupLE :: Int -> IntSet -> Maybe Int
intsetLookupLE = IntSet.lookupLE

intsetLookupLT :: Int -> IntSet -> Maybe Int
intsetLookupLT = IntSet.lookupLT

intsetMap :: (Int -> Int) -> IntSet -> IntSet
intsetMap = IntSet.map

intsetMapMonotonic :: (Int -> Int) -> IntSet -> IntSet
intsetMapMonotonic = IntSet.mapMonotonic

intsetMaxView :: IntSet -> Maybe (Int, IntSet)
intsetMaxView = IntSet.maxView

intsetMember :: Int -> IntSet -> Bool
intsetMember = IntSet.member

intsetMinView :: IntSet -> Maybe (Int, IntSet)
intsetMinView = IntSet.minView

intsetNotMember :: Int -> IntSet -> Bool
intsetNotMember = IntSet.notMember

intsetNull :: IntSet -> Bool
intsetNull = IntSet.null

intsetPartition :: (Int -> Bool) -> IntSet -> (IntSet, IntSet)
intsetPartition = IntSet.partition

intsetShowTree :: IntSet -> String
intsetShowTree = IntSet.showTree

intsetShowTreeWith :: Bool -> Bool -> IntSet -> String
intsetShowTreeWith = IntSet.showTreeWith

intsetSingleton :: Int -> IntSet
intsetSingleton = IntSet.singleton

intsetSize :: IntSet -> Int
intsetSize = IntSet.size

intsetSplit :: Int -> IntSet -> (IntSet, IntSet)
intsetSplit = IntSet.split

intsetSplitMember :: Int -> IntSet -> (IntSet, Bool, IntSet)
intsetSplitMember = IntSet.splitMember

intsetSplitRoot :: IntSet -> [IntSet]
intsetSplitRoot = IntSet.splitRoot

intsetToAscList :: IntSet -> [Int]
intsetToAscList = IntSet.toAscList

intsetToDescList :: IntSet -> [Int]
intsetToDescList = IntSet.toDescList

intsetToList :: IntSet -> [Int]
intsetToList = IntSet.toList

intsetUnion :: IntSet -> IntSet -> IntSet
intsetUnion = IntSet.union

intsetUnions :: Foldable f => f IntSet -> IntSet
intsetUnions = IntSet.unions

--
--
--

type IntMap = IM.IntMap

imAdjust :: (a -> a) -> Int -> IntMap a -> IntMap a
imAdjust = IM.adjust

imAdjustWithKey :: (Int -> a -> a) -> Int -> IntMap a -> IntMap a
imAdjustWithKey = IM.adjustWithKey

imAlter :: (Maybe a -> Maybe a) -> Int -> IntMap a -> IntMap a
imAlter = IM.alter

imAlterF :: Functor f => (Maybe a -> f (Maybe a)) -> Int -> IntMap a -> f (IntMap a)
imAlterF = IM.alterF

imAssocs :: IntMap a -> [(Int, a)]
imAssocs = IM.assocs

imCompose :: IntMap c -> IntMap Int -> IntMap c
imCompose = IM.compose

imDelete :: Int -> IntMap a -> IntMap a
imDelete = IM.delete

imDeleteFindMax :: IntMap a -> ((Int, a), IntMap a)
imDeleteFindMax = IM.deleteFindMax

imDeleteFindMin :: IntMap a -> ((Int, a), IntMap a)
imDeleteFindMin = IM.deleteFindMin

imDeleteMax :: IntMap a -> IntMap a
imDeleteMax = IM.deleteMax

imDeleteMin :: IntMap a -> IntMap a
imDeleteMin = IM.deleteMin

imDifference :: IntMap a -> IntMap b -> IntMap a
imDifference = IM.difference

imDifferenceWith :: (a -> b -> Maybe a) -> IntMap a -> IntMap b -> IntMap a
imDifferenceWith = IM.differenceWith

imDifferenceWithKey :: (Int -> a -> b -> Maybe a) -> IntMap a -> IntMap b -> IntMap a
imDifferenceWithKey = IM.differenceWithKey

imDisjoint :: IntMap a -> IntMap b -> Bool
imDisjoint = IM.disjoint

imElems :: IntMap a -> [a]
imElems = IM.elems

imEmpty :: IntMap a
imEmpty = IM.empty

imFilter :: (a -> Bool) -> IntMap a -> IntMap a
imFilter = IM.filter

imFilterWithKey :: (Int -> a -> Bool) -> IntMap a -> IntMap a
imFilterWithKey = IM.filterWithKey

imFindMax :: IntMap a -> (Int, a)
imFindMax = IM.findMax

imFindMin :: IntMap a -> (Int, a)
imFindMin = IM.findMin

imFindWithDefault :: a -> Int -> IntMap a -> a
imFindWithDefault = IM.findWithDefault

imFoldMapWithKey :: Monoid m => (Int -> a -> m) -> IntMap a -> m
imFoldMapWithKey = IM.foldMapWithKey

imFoldl :: (a -> b -> a) -> a -> IntMap b -> a
imFoldl = IM.foldl

imFoldl' :: (a -> b -> a) -> a -> IntMap b -> a
imFoldl' = IM.foldl'

imFoldlWithKey :: (a -> Int -> b -> a) -> a -> IntMap b -> a
imFoldlWithKey = IM.foldlWithKey

imFoldlWithKey' :: (a -> Int -> b -> a) -> a -> IntMap b -> a
imFoldlWithKey' = IM.foldlWithKey'

imFoldr :: (a -> b -> b) -> b -> IntMap a -> b
imFoldr = IM.foldr

imFoldr' :: (a -> b -> b) -> b -> IntMap a -> b
imFoldr' = IM.foldr'

imFoldrWithKey :: (Int -> a -> b -> b) -> b -> IntMap a -> b
imFoldrWithKey = IM.foldrWithKey

imFoldrWithKey' :: (Int -> a -> b -> b) -> b -> IntMap a -> b
imFoldrWithKey' = IM.foldrWithKey'

imFromAscList :: [(Int, a)] -> IntMap a
imFromAscList = IM.fromAscList

imFromAscListWith :: (a -> a -> a) -> [(Int, a)] -> IntMap a
imFromAscListWith = IM.fromAscListWith

imFromAscListWithKey :: (Int -> a -> a -> a) -> [(Int, a)] -> IntMap a
imFromAscListWithKey = IM.fromAscListWithKey

imFromDistinctAscList :: [(Int, a)] -> IntMap a
imFromDistinctAscList = IM.fromDistinctAscList

imFromList :: [(Int, a)] -> IntMap a
imFromList = IM.fromList

imFromListWith :: (a -> a -> a) -> [(Int, a)] -> IntMap a
imFromListWith = IM.fromListWith

imFromListWithKey :: (Int -> a -> a -> a) -> [(Int, a)] -> IntMap a
imFromListWithKey = IM.fromListWithKey

imFromSet :: (Int -> a) -> IntSet -> IntMap a
imFromSet = IM.fromSet

imInsert :: Int -> a -> IntMap a -> IntMap a
imInsert = IM.insert

imInsertLookupWithKey :: (Int -> a -> a -> a) -> Int -> a -> IntMap a -> (Maybe a, IntMap a)
imInsertLookupWithKey = IM.insertLookupWithKey

imInsertWith :: (a -> a -> a) -> Int -> a -> IntMap a -> IntMap a
imInsertWith = IM.insertWith

imInsertWithKey :: (Int -> a -> a -> a) -> Int -> a -> IntMap a -> IntMap a
imInsertWithKey = IM.insertWithKey

imIntersection :: IntMap a -> IntMap b -> IntMap a
imIntersection = IM.intersection

imIntersectionWith :: (a -> b -> c) -> IntMap a -> IntMap b -> IntMap c
imIntersectionWith = IM.intersectionWith

imIntersectionWithKey :: (Int -> a -> b -> c) -> IntMap a -> IntMap b -> IntMap c
imIntersectionWithKey = IM.intersectionWithKey

imIsProperSubmapOf :: Eq a => IntMap a -> IntMap a -> Bool
imIsProperSubmapOf = IM.isProperSubmapOf

imIsProperSubmapOfBy :: (a -> b -> Bool) -> IntMap a -> IntMap b -> Bool
imIsProperSubmapOfBy = IM.isProperSubmapOfBy

imIsSubmapOf :: Eq a => IntMap a -> IntMap a -> Bool
imIsSubmapOf = IM.isSubmapOf

imIsSubmapOfBy :: (a -> b -> Bool) -> IntMap a -> IntMap b -> Bool
imIsSubmapOfBy = IM.isSubmapOfBy

imKeys :: IntMap a -> [Int]
imKeys = IM.keys

imKeysSet :: IntMap a -> IntSet
imKeysSet = IM.keysSet

imLookup :: Int -> IntMap a -> Maybe a
imLookup = IM.lookup

imLookupGE :: Int -> IntMap a -> Maybe (Int, a)
imLookupGE = IM.lookupGE

imLookupGT :: Int -> IntMap a -> Maybe (Int, a)
imLookupGT = IM.lookupGT

imLookupLE :: Int -> IntMap a -> Maybe (Int, a)
imLookupLE = IM.lookupLE

imLookupLT :: Int -> IntMap a -> Maybe (Int, a)
imLookupLT = IM.lookupLT

imLookupMax :: IntMap a -> Maybe (Int, a)
imLookupMax = IM.lookupMax

imLookupMin :: IntMap a -> Maybe (Int, a)
imLookupMin = IM.lookupMin

imMap :: (a -> b) -> IntMap a -> IntMap b
imMap = IM.map

imMapAccum :: (a -> b -> (a, c)) -> a -> IntMap b -> (a, IntMap c)
imMapAccum = IM.mapAccum

imMapAccumRWithKey :: (a -> Int -> b -> (a, c)) -> a -> IntMap b -> (a, IntMap c)
imMapAccumRWithKey = IM.mapAccumRWithKey

imMapAccumWithKey :: (a -> Int -> b -> (a, c)) -> a -> IntMap b -> (a, IntMap c)
imMapAccumWithKey = IM.mapAccumWithKey

imMapEither :: (a -> Either b c) -> IntMap a -> (IntMap b, IntMap c)
imMapEither = IM.mapEither

imMapEitherWithKey :: (Int -> a -> Either b c) -> IntMap a -> (IntMap b, IntMap c)
imMapEitherWithKey = IM.mapEitherWithKey

imMapKeys :: (Int -> Int) -> IntMap a -> IntMap a
imMapKeys = IM.mapKeys

imMapKeysMonotonic :: (Int -> Int) -> IntMap a -> IntMap a
imMapKeysMonotonic = IM.mapKeysMonotonic

imMapKeysWith :: (a -> a -> a) -> (Int -> Int) -> IntMap a -> IntMap a
imMapKeysWith = IM.mapKeysWith

imMapMaybe :: (a -> Maybe b) -> IntMap a -> IntMap b
imMapMaybe = IM.mapMaybe

imMapMaybeWithKey :: (Int -> a -> Maybe b) -> IntMap a -> IntMap b
imMapMaybeWithKey = IM.mapMaybeWithKey

imMapWithKey :: (Int -> a -> b) -> IntMap a -> IntMap b
imMapWithKey = IM.mapWithKey

imMaxView :: IntMap a -> Maybe (a, IntMap a)
imMaxView = IM.maxView

imMaxViewWithKey :: IntMap a -> Maybe ((Int, a), IntMap a)
imMaxViewWithKey = IM.maxViewWithKey

imMember :: Int -> IntMap a -> Bool
imMember = IM.member

imMergeWithKey :: (Int -> a -> b -> Maybe c) -> (IntMap a -> IntMap c) -> (IntMap b -> IntMap c) -> IntMap a -> IntMap b -> IntMap c
imMergeWithKey = IM.mergeWithKey

imMinView :: IntMap a -> Maybe (a, IntMap a)
imMinView = IM.minView

imMinViewWithKey :: IntMap a -> Maybe ((Int, a), IntMap a)
imMinViewWithKey = IM.minViewWithKey

imNotMember :: Int -> IntMap a -> Bool
imNotMember = IM.notMember

imNull :: IntMap a -> Bool
imNull = IM.null

imPartition :: (a -> Bool) -> IntMap a -> (IntMap a, IntMap a)
imPartition = IM.partition

imPartitionWithKey :: (Int -> a -> Bool) -> IntMap a -> (IntMap a, IntMap a)
imPartitionWithKey = IM.partitionWithKey

imRestrictKeys :: IntMap a -> IntSet -> IntMap a
imRestrictKeys = IM.restrictKeys

imSingleton :: Int -> a -> IntMap a
imSingleton = IM.singleton

imSize :: IntMap a -> Int
imSize = IM.size

imSplit :: Int -> IntMap a -> (IntMap a, IntMap a)
imSplit = IM.split

imSplitLookup :: Int -> IntMap a -> (IntMap a, Maybe a, IntMap a)
imSplitLookup = IM.splitLookup

imSplitRoot :: IntMap a -> [IntMap a]
imSplitRoot = IM.splitRoot

imToAscList :: IntMap a -> [(Int, a)]
imToAscList = IM.toAscList

imToDescList :: IntMap a -> [(Int, a)]
imToDescList = IM.toDescList

imToList :: IntMap a -> [(Int, a)]
imToList = IM.toList

imTraverseMaybeWithKey :: Applicative f => (Int -> a -> f (Maybe b)) -> IntMap a -> f (IntMap b)
imTraverseMaybeWithKey = IM.traverseMaybeWithKey

imTraverseWithKey :: Applicative t => (Int -> a -> t b) -> IntMap a -> t (IntMap b)
imTraverseWithKey = IM.traverseWithKey

imUnion :: IntMap a -> IntMap a -> IntMap a
imUnion = IM.union

imUnionWith :: (a -> a -> a) -> IntMap a -> IntMap a -> IntMap a
imUnionWith = IM.unionWith

imUnionWithKey :: (Int -> a -> a -> a) -> IntMap a -> IntMap a -> IntMap a
imUnionWithKey = IM.unionWithKey

imUnions :: Foldable f => f (IntMap a) -> IntMap a
imUnions = IM.unions

imUnionsWith :: Foldable f => (a -> a -> a) -> f (IntMap a) -> IntMap a
imUnionsWith = IM.unionsWith

imUpdate :: (a -> Maybe a) -> Int -> IntMap a -> IntMap a
imUpdate = IM.update

imUpdateLookupWithKey :: (Int -> a -> Maybe a) -> Int -> IntMap a -> (Maybe a, IntMap a)
imUpdateLookupWithKey = IM.updateLookupWithKey

imUpdateMax :: (a -> Maybe a) -> IntMap a -> IntMap a
imUpdateMax = IM.updateMax

imUpdateMaxWithKey :: (Int -> a -> Maybe a) -> IntMap a -> IntMap a
imUpdateMaxWithKey = IM.updateMaxWithKey

imUpdateMin :: (a -> Maybe a) -> IntMap a -> IntMap a
imUpdateMin = IM.updateMin

imUpdateMinWithKey :: (Int -> a -> Maybe a) -> IntMap a -> IntMap a
imUpdateMinWithKey = IM.updateMinWithKey

imUpdateWithKey :: (Int -> a -> Maybe a) -> Int -> IntMap a -> IntMap a
imUpdateWithKey = IM.updateWithKey

imWithoutKeys :: IntMap a -> IntSet -> IntMap a
imWithoutKeys = IM.withoutKeys
