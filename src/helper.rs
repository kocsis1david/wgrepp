use std::ops::{Rem, Sub};

use num::Zero;

pub fn align_to<S>(x: S, alignment: S) -> S
where
    S: Copy + Sub<Output = S> + Rem<Output = S> + PartialEq + Zero,
{
    let r = x % alignment;
    if r != S::zero() {
        x - r
    } else {
        x
    }
}
