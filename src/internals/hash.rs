use std::hash::Hasher;

/// A hasher optimized for hashing component type IDs.
#[derive(Default)]
pub struct ComponentTypeIdHasher(u64);

impl Hasher for ComponentTypeIdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write_u64(&mut self, seed: u64) {
        // This must only be used to hash one value.
        debug_assert_eq!(self.0, 0);
        self.0 = seed;
    }

    fn write(&mut self, _bytes: &[u8]) {
        // This should not be called, only write_u64.
        unimplemented!()
    }
}

/// A hasher optimized for hashing types that are represented as a u64.
#[derive(Default)]
pub struct U64Hasher(u64);

impl Hasher for U64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write_u64(&mut self, seed: u64) {
        // This must only be used to hash one value.
        debug_assert_eq!(self.0, 0);

        let max_prime = 11_400_714_819_323_198_549u64;
        self.0 = max_prime.wrapping_mul(seed);
    }

    fn write(&mut self, _bytes: &[u8]) {
        // This should not be called, only write_u64.
        unimplemented!()
    }
}

#[test]
fn hasher() {
    fn verify<T: 'static + ?Sized>() {
        use core::{any::TypeId, hash::Hash};

        // hashing a type ID must go through `write_u64` (not `write`, which is
        // unimplemented) and must be deterministic
        let mut hasher = ComponentTypeIdHasher::default();
        TypeId::of::<T>().hash(&mut hasher);
        let first = hasher.finish();

        let mut hasher = ComponentTypeIdHasher::default();
        TypeId::of::<T>().hash(&mut hasher);
        assert_eq!(first, hasher.finish());
    }

    verify::<usize>();
    verify::<()>();
    verify::<str>();
    verify::<&'static str>();
    verify::<[u8; 20]>();

    // distinct types hash to distinct values
    fn hash_of<T: 'static>() -> u64 {
        use core::{any::TypeId, hash::Hash};
        let mut hasher = ComponentTypeIdHasher::default();
        TypeId::of::<T>().hash(&mut hasher);
        hasher.finish()
    }
    assert_ne!(hash_of::<usize>(), hash_of::<u8>());
}
