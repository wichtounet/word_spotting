//=======================================================================
// Copyright Baptiste Wicht 2015-2017.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef WORD_SPOTTER_DTW_HPP
#define WORD_SPOTTER_DTW_HPP

template <typename V1, typename V2>
double dtw_distance(const V1& s, const V2& t, bool sc_band = true, double band = 0.1) {
    const auto n = s.size();
    const auto m = t.size();

    auto ratio = static_cast<double>(n) / m;
    if (ratio > 2.0 || ratio < 0.5) {
        return 100000.0;
    }

    auto d = [&s, &t](std::size_t i, std::size_t j) { return std::sqrt(etl::sum((s[i] - t[j]) >> (s[i] - t[j]))); };

    etl::dyn_matrix<double> dtw(n, m);

    dtw(0, 0) = d(0, 0);

    for (std::size_t i = 1; i < n; i++) {
        dtw(i, 0) = d(i, 0) + dtw(i - 1, 0);
    }

    for (std::size_t j = 1; j < m; j++) {
        dtw(0, j) = d(0, j) + dtw(0, j - 1);
    }

    for (std::size_t i = 1; i < n; i++) {
        for (std::size_t j = 1; j < m; j++) {
            //Sakoe-Chiba constraint
            if (sc_band && (j < (m * static_cast<double>(i) / n) - band * m || j > (m * static_cast<double>(i) / n) + band * m)) {
                dtw(i, j) = 100000.0;
                continue;
            }

            dtw(i, j) = d(i, j) + std::min(dtw(i - 1, j), std::min(dtw(i - 1, j - 1), dtw(i, j - 1)));
        }
    }

    std::size_t i = n - 1;
    std::size_t j = m - 1;
    std::size_t K = 1;

    while ((i + j) > 0) {
        if (i == 0) {
            --j;
        } else if (j == 0) {
            --i;
        } else {
            if (dtw(i - 1, j - 1) < dtw(i - 1, j) && dtw(i - 1, j - 1) < dtw(i, j - 1)) {
                --i;
                --j;
            } else if (dtw(i - 1, j) < dtw(i, j - 1)) {
                --i;
            } else {
                --j;
            }
        }

        ++K;
    }

    return dtw(n - 1, m - 1) / K;
}

#endif
