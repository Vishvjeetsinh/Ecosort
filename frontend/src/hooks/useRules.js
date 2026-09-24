import { useCallback, useEffect, useState } from 'react';
import * as api from '../lib/api.js';

function isAbort(err) {
  return Boolean(err) && err.name === 'AbortError';
}

/**
 * Loads the category taxonomy, the region catalogue and the selected region's rules.
 *
 * The catalogue is fetched once; the per-region rule set is refetched whenever
 * `regionId` changes, aborting whatever request is still in flight so a fast region
 * switch can never land an older response on top of a newer one.
 *
 * @param {string} regionId  falsy until the app has resolved a default region
 */
export default function useRules(regionId) {
  const [catalog, setCatalog] = useState({
    categories: [],
    regions: [],
    defaultRegion: null,
  });
  const [rules, setRules] = useState(null);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [rulesLoading, setRulesLoading] = useState(false);
  const [error, setError] = useState(null);
  const [reloadToken, setReloadToken] = useState(0);

  const reload = useCallback(() => {
    setError(null);
    setReloadToken((n) => n + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    setCatalogLoading(true);
    Promise.all([
      api.getCategories(controller.signal),
      api.getRules(controller.signal),
    ])
      .then(([categoriesResponse, rulesResponse]) => {
        if (!active) return;
        setCatalog({
          categories: Array.isArray(categoriesResponse?.categories)
            ? categoriesResponse.categories
            : [],
          regions: Array.isArray(rulesResponse?.regions) ? rulesResponse.regions : [],
          defaultRegion: rulesResponse?.defaultRegion || null,
        });
        setError(null);
      })
      .catch((err) => {
        if (!active || isAbort(err)) return;
        setError(err);
      })
      .finally(() => {
        if (active) setCatalogLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [reloadToken]);

  useEffect(() => {
    if (!regionId) {
      setRules(null);
      setRulesLoading(false);
      return undefined;
    }

    const controller = new AbortController();
    let active = true;

    setRulesLoading(true);
    api
      .getRegionRules(regionId, controller.signal)
      .then((response) => {
        if (!active) return;
        setRules(response || null);
        setError(null);
      })
      .catch((err) => {
        if (!active || isAbort(err)) return;
        setRules(null);
        setError(err);
      })
      .finally(() => {
        if (active) setRulesLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [regionId, reloadToken]);

  return {
    categories: catalog.categories,
    regions: catalog.regions,
    defaultRegion: catalog.defaultRegion,
    rules,
    loading: catalogLoading || rulesLoading,
    error,
    reload,
  };
}

export { useRules };
